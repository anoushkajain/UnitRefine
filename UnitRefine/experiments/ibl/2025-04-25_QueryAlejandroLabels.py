# %%
import tqdm
from one.api import ONE
from pathlib import Path
import pandas as pd
one = ONE(base_url='https://alyx.internationalbrainlab.org')

dsets = one.alyx.rest('datasets', 'list', django=
'name__icontains,clusters.curatedLabels,session__projects__name__istartswith,witten'
                      )

df_recordings = pd.DataFrame(
    {'eid': [dset['session'][-36:] for dset in dsets],
     'pname': [dset['collection'].split('/')[1] for dset in dsets],
    }
)
df_recordings['pid'] = [str(one.eid2pid(eid=rec.eid, name=rec.pname)[0][0]) for _, rec in df_recordings.iterrows()]


repo_path = Path('/Users/olivier/Documents/PYTHON/00_IBL/UnitRefine/UnitRefine/experiments/ibl')
df_recordings.to_csv(repo_path.joinpath('insertions_with_manual_curation.csv'), index=False)


# %%
from brainbox.io.one import SpikeSortingLoader
all_clusters = []
for i, rec in tqdm.tqdm(df_recordings.iterrows(), total=df_recordings.shape[0]):
    ssl = SpikeSortingLoader(pid=rec['pid'], one=one)
    clusters = ssl.load_spike_sorting_object('clusters', namespace='')
    channels = ssl.load_channels()
    df_clusters = pd.DataFrame(ssl.merge_clusters(None, clusters, channels))

    dset = one.list_datasets(eid=rec['eid'], filename='_av_clusters.curatedLabels*', collection=ssl.collection)
    curated_labels = one.load_dataset(id=rec['eid'], dataset=dset[0])
    df_clusters['curated_labels'] = curated_labels['group']
    df_clusters['pid'] = rec['pid']
    all_clusters.append(df_clusters)


pd.concat(all_clusters).to_parquet('/Users/olivier/Documents/datadisk/unitrefine/alejandro/all_units.pqt')
