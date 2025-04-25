from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np
import pandas as pd

from ibl_style.style import figure_style
figure_style()
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

# %% Step 1: load data and the IBL metrics
one = ONE(base_url='https://alyx.internationalbrainlab.org')
pid = 'fe380793-8035-414e-b000-09bfe5ece92a'
ssl = SpikeSortingLoader(one=one, pid=pid)
clusters = ssl.load_spike_sorting_object('clusters')


# %% Step 2: load the Unit Refine labels
OUT_PATH = Path('/Users/olivier/Documents/scratch/brainhack')
df_unit_refine = pd.read_parquet(Path.home().joinpath(OUT_PATH.joinpath(f'{pid}_unit_refine.pqt')))

# %% Step 3: run the sirena model
import sirena.classifier

path_terermon = Path('/media/ccu-storage/teremon')
files_manual_curation = [
    # Ferrero Rocher: S6  incomplete, only imec1 (striatum),
    '/media/ccu-storage/shared-paton/teremon/ephys_curated/20230801_ChocolateGroup/5_FerreroRocher/20082023_Ferrero_StrCer_S6_g0/20082023_Ferrero_StrCer_S6_g0_imec1/catGT/kilosort4/cluster_group.tsv',
    # The IBL sort
    '/media/ccu-storage/shared-paton/teremon/ephys_curated/20230801_ChocolateGroup/4_Milka/18082023_Milka_StrCer_S4_g0/18082023_Milka_StrCer_S4_g0_imec1/ibl_sorter_results_default/cluster_group.tsv',
]

model_path = Path(sirena.__file__).parents[2].joinpath('models')
df_probas, _ = sirena.classifier.predict(model_path, df_metrics=clusters['metrics'])
pred_sirena = np.array(['sua', 'mua', 'noise'])[np.argmax(df_probas.values, axis=1)]

OUT_PATH = Path('/Users/olivier/Documents/scratch/brainhack')
np.save(OUT_PATH.joinpath('sirena.npy'), pred_sirena)



fail_ibl = clusters['metrics']['bitwise_fail'] > 0
fail_ure = df_unit_refine['prediction'].values != 'sua'
fail_sir = pred_sirena != 'sua'


# %%
# For a 2-set Venn diagram

fig, axs = plt.subplots(ncols=3, nrows=3)
import addcopyfighandler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def plot_comp(ax, fail0, fail1, labels):
    venn2((set(np.where(fail0)[0]), set(np.where(fail1)[0])), set_labels = labels, ax=ax[0])
    venn2((set(np.where(~fail0)[0]), set(np.where(~fail1)[0])), set_labels = labels, ax=ax[1])
    acc = accuracy_score(fail0, fail1)
    cm = confusion_matrix(fail0, fail1, normalize='all')
    cm_df = pd.DataFrame(cm,
                         index=[f'{labels[0]}:good', f'{labels[0]}:noise'],
                         columns=[f'{labels[1]}:good', f'{labels[1]}:noise'])
    sns.heatmap(cm_df, annot=True, cmap='Blues', ax=ax[2])
    plt.title('Confusion Matrix')
    plt.ylabel('Sirena Label')
    plt.xlabel('Unit Refine Label')
    plt.show()
    ax[0].set(title='Overall Agreement: {:.2f}'.format(acc))
    ax[1].set(title='Overall Agreement: {:.2f}'.format(acc))

plot_comp(axs[0], fail_ibl, fail_ure, ['IBL', 'UnitRefine'])
plot_comp(axs[1], fail_ibl, fail_sir, ['IBL', 'Sirena'])
plot_comp(axs[2], fail_ure, fail_sir, ['UnitRefine', 'Sirena'])
