# %%
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

# here we use public BWM datasets
path_model = Path('/Users/olivier/Documents/datadisk/unitrefine/alejandro/AlejandroPanVazquez_IBL_slim-mushroom-alligator')
one = ONE(base_url='https://openalyx.internationalbrainlab.org')
eid = 'fa1f26a1-eb49-4b24-917e-19f02a18ac61'
pids, pnames = one.eid2pid(eid)
pid = pids[0]
ssl = SpikeSortingLoader(one=one, pid=pid)
clusters = ssl.load_spike_sorting_object('clusters', namespace='')
channels = ssl.load_channels()
df_clusters = pd.DataFrame(ssl.merge_clusters(None, clusters, channels))

# %%

def venn_units(i_ibl, i_ur):
    num_ibl = np.sum(i_ibl)
    num_cur = np.sum(i_ur)
    num_both = np.sum(i_ibl & i_ur)

    # Create a dictionary for the Venn diagram
    venn_dict = {
        "11": num_both,                # Both IBL and curator
        "10": num_ibl - num_both,      # Only IBL
        "01": num_cur - num_both       # Only curator
    }
    # Create the Venn diagram
    plt.figure(figsize=(10, 6))
    venn2(subsets=venn_dict, set_labels=('IBL', 'Curator'))
    plt.title("Overlap between IBL and Curator selections")

    # Add counts as text for clarity
    total_units = len(i_ibl)
    plt.figtext(0.5, 0.01, f"Total units: {total_units}", ha="center")
    plt.figtext(0.5, 0.05, f"IBL: {num_ibl}, Curator: {num_cur}, Both: {num_both}", ha="center")

    plt.tight_layout()
    plt.show()


def load_model(path_model):
    path_model = Path(path_model)
    with open(path_model.joinpath("meta.yaml")) as f:
        metadata = yaml.safe_load(f)
    classifier = XGBClassifier(model_file=path_model.joinpath("model.ubj"))
    classifier.load_model(path_model.joinpath("model.ubj"))
    return classifier, metadata

classifier, metadata = load_model(path_model)

classifier.predict_proba(df_clusters.loc[:, metadata['FEATURES']])
pred = classifier.predict(df_clusters.loc[:, metadata['FEATURES']])

# %%
import addcopyfighandler
import matplotlib
matplotlib.use('qtagg')
plt.style.use('default')
venn_units(df_clusters['bitwise_fail'].values == 0, i_ur = pred == 1)

