# %%
from one.api import ONE
from pathlib import Path
import pandas as pd
one = ONE(base_url='https://alyx.internationalbrainlab.org')

dsets = one.alyx.rest('datasets', 'list', django='name__icontains,clusters.curatedLabels')

df_recordings = pd.DataFrame(
    {'eid': [dset['session'][-36:] for dset in dsets],
     'pname': [dset['collection'].split('/')[1] for dset in dsets],
    }
)
df_recordings['pid'] = [str(one.eid2pid(eid=rec.eid, name=rec.pname)[0][0]) for _, rec in df_recordings.iterrows()]


repo_path = Path('/Users/olivier/Documents/PYTHON/00_IBL/UnitRefine/UnitRefine/experiments/ibl')
df_recordings.to_csv(repo_path.joinpath('insertions_with_manual_curation.csv'), index=False)
