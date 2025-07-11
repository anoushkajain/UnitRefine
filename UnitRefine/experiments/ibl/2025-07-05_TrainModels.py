# %%
import hashlib
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import sklearn.metrics
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

import iblutil.random
import iblutil.numerical

unit_refine_folder = Path('/Users/olivier/Documents/datadisk/unitrefine/alejandro')
df_clusters = pd.read_parquet(unit_refine_folder.joinpath('all_units.pqt'))
df_clusters = df_clusters[df_clusters['curated_labels'] != 'unsorted']
df_clusters['curated_labels_binary'] = 'bad'
df_clusters.loc[df_clusters['curated_labels'] == 'good', 'curated_labels_binary'] = 'good'


def save_model(path_model, classifier, meta, subfolder="", identifier=None):
    """
    Save model to disk in ubj format with associated meta-data and a hash
    The model is a set of files in a folder named after the meta-data
     'VINTAGE' and 'REGION_MAP' fields, with the hash as suffix e.g. 2023_W41_Cosmos_dfd731f0
    :param classifier:
    :param meta:
    :param path_model:
    :param identifier: optional identifier for the model, defaults to a 8 character hexdigest of the meta data
    :param subfolder: optional level to add to the model path, for example 'FOLD01' will write to
        2023_W41_Cosmos_dfd731f0/FOLD01/
    :return:
    """
    meta["MODEL_CLASS"] = (
        f"{classifier.__class__.__module__}.{classifier.__class__.__name__}"
    )
    if identifier is None:
        identifier = iblutil.random.name_from_hash(
            hashlib.md5(yaml.dump(meta).encode("utf-8"))
        )
    path_model = path_model.joinpath(
        f"{meta['DATASET']}_{meta['FEATURES_TYPE']}_{identifier}", subfolder
    )
    path_model.mkdir(exist_ok=True, parents=True)
    with open(path_model.joinpath("meta.yaml"), "w+") as fid:
        fid.write(yaml.dump(dict(meta)))
    classifier.save_model(path_model.joinpath("model.ubj"))
    return path_model


def train_fold(df_clusters, test_idx, fold_label='', x_list=None, train_label=None, classes=None):
    train_idx = ~test_idx
    print(f"{fold_label}: {df_clusters.shape[0]} channels", f'training set {np.sum(test_idx) / test_idx.size}')
    x_train = df_clusters.loc[train_idx, x_list].values
    x_test = df_clusters.loc[test_idx, x_list].values
    y_train = df_clusters.loc[train_idx, train_label].values
    y_test = df_clusters.loc[test_idx, train_label].values
    df_test = df_clusters.loc[test_idx, :].copy()
    _, iy_train = iblutil.numerical.ismember(y_train, classes)
    _, iy_test = iblutil.numerical.ismember(y_test, classes)
    classifier = XGBClassifier(device="gpu", verbosity=2)
    # fit model
    classifier.fit(x_train, iy_train)
    # make predictions
    y_pred = classes[classifier.predict(x_test)]
    df_test[f"cosmos_prediction"] = classes[
        classifier.predict(df_test.loc[:, x_list].values)
    ]
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_test, y_pred, normalize="true"
    )  # row: true, col: predicted
    print(f"{fold_label} Accuracy: {accuracy}")
    return classifier.predict_proba(x_test), classifier, accuracy, confusion_matrix

# %%
n_folds = 5
x_list = ['amp_max', 'amp_min',
       'amp_median', 'amp_std_dB', 'contamination', 'contamination_alt',
       'drift', 'missed_spikes_est', 'noise_cutoff', 'presence_ratio',
       'presence_ratio_std', 'slidingRP_viol', 'spike_count',
       'slidingRP_viol_forced', 'max_confidence', 'min_contamination',
       'n_spikes_below2', 'firing_rate']
TRAIN_LABEL = 'curated_labels_binary'
nc = df_clusters.shape[0]
rs = np.random.seed(12345)
ifold = np.floor(np.arange(nc) / nc * n_folds)
np.random.shuffle(ifold)
classes = np.unique(df_clusters[TRAIN_LABEL].values)
print(df_clusters[TRAIN_LABEL].value_counts())
print(df_clusters[TRAIN_LABEL].value_counts() / nc)

confusion_matrix = np.zeros((len(classes), len(classes)))
df_predictions = pd.DataFrame(index=df_clusters.index, columns=list(classes) + ['prediction', 'fold'], dtype=float)
for i in range(n_folds):
    test_idx = ifold == i
    probas, classifier, accuracy, _confusion_matrix = train_fold(
        df_clusters,
        test_idx=test_idx,
        fold_label=f"fold {i}",
        x_list=x_list,
        train_label=TRAIN_LABEL,
        classes=classes
    )
    confusion_matrix += _confusion_matrix / n_folds
    df_predictions.loc[test_idx, classes] = probas
    df_predictions.loc[test_idx, 'fold'] = i
    df_predictions.loc[test_idx, 'prediction'] = classes[np.argmax(probas, axis=1)]

accuracy = sklearn.metrics.accuracy_score(df_clusters[TRAIN_LABEL].values, df_predictions['prediction'].values.astype(classes.dtype))
print(f"Overall Accuracy: {accuracy}")
balanced_accuracy = sklearn.metrics.balanced_accuracy_score(df_clusters[TRAIN_LABEL].values, df_predictions['prediction'].values.astype(classes.dtype))
print(f"Overall Balanced Accuracy: {balanced_accuracy}")


meta = dict(
    DATASET='AlejandroPanVazquez',
    RANDOM_SEED=rs,
    FEATURES_TYPE="IBL",
    FEATURES=x_list,
    CLASSES=list(classes),
    ACCURACY=accuracy,
    TRAINING=dict(
        training_size=nc,
        testing_size=nc / n_folds,
        hash_training=iblutil.numerical.hash_uuids(df_clusters['uuids']),
    ),
)

#
# path_model = save_model(unit_refine_folder, classifier, meta)
# print(f"Saved model to {path_model}")

# %%

# Plot the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
ax = sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    vmin=0,
    vmax=1
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.3f}, Balanced Accuracy: {balanced_accuracy:.3f}')

