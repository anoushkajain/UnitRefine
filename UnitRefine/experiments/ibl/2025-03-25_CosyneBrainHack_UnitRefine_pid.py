# %%
import time
import numpy as np


from spikeinterface.extractors import IblRecordingExtractor, IblSortingExtractor
import spikeinterface.core.analyzer_extension_core as extensions
from spikeinterface.postprocessing.spike_amplitudes import ComputeSpikeAmplitudes
import spikeinterface.preprocessing as spre

import spikeinterface as si
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

tstart = time.time()

print('Load recording')
one = ONE(base_url='https://alyx.internationalbrainlab.org')
pid = 'fe380793-8035-414e-b000-09bfe5ece92a'

ssl = SpikeSortingLoader(one=one, pid=pid)
wl = ssl.raw_waveforms()  # TODO: deactivate checksum computation
# %load_ext autoreload
# %autoreload 2
recording = IblRecordingExtractor(one=one, pid=pid, stream_type='ap')
sorting = IblSortingExtractor(one=one, pid=pid)
recording = spre.highpass_filter(recording, 300)
recording = spre.phase_shift(recording)
recording = spre.common_reference(recording)


# /Users/olivier/Documents/datadisk/scratch/Subjects/KM_012/2024-03-05/002/raw_ephys_data/probe00
# %%
print(f'Get full templates {time.time() - tstart}')

j = wl.channels[wl.df_clusters['first_index'], :]
i = np.tile(np.arange(wl.templates.shape[0]).astype((np.int16))[:, np.newaxis], (1, wl.nc))
sparsity = np.zeros((wl.templates.shape[0], recording.get_num_channels() + 1), dtype=bool)
sparsity[i.flat, j.flat] = True
sparsity = sparsity[:, :-1]

full_templates = np.zeros((wl.nu, wl.ns, recording.get_num_channels() + 1), dtype=np.float32)
for ic in np.arange(wl.nu):
    ichannels = wl.channels[wl.df_clusters['first_index'].values[ic], :]
    full_templates[ic, :, ichannels] = wl.templates[ic, :, :]
full_templates = full_templates[:, :, :-1]

# %%
print(f'Instantiate analyzer {time.time() - tstart}')
channel_sparsity = si.ChannelSparsity(sparsity, unit_ids=sorting.get_unit_ids(), channel_ids=recording.get_channel_ids())
analyzer = si.create_sorting_analyzer(sorting, recording, sparsity=channel_sparsity)  # this takes around 30 seconds

# %%
print(f'Set analyzer extensions {time.time() - tstart}')
# TODO need to find the original spike index using a chunked approach
ext = extensions.ComputeRandomSpikes(analyzer)
ext.set_params()
ext.data['random_spikes_indices'] = np.arange(sorting.to_spike_vector().size)
ext.run_info = {"run_completed": True}
analyzer.extensions['random_spikes'] = ext

ect = extensions.ComputeTemplates(analyzer)
# analyzer.extensions['templates'] =
ect.set_params(
    ms_before=wl.trough_offset / analyzer.sampling_frequency * 1000,
    ms_after=(wl.ns - wl.trough_offset) / analyzer.sampling_frequency * 1000,
    operators=['average', 'std'],  # todo check if this is indeed the case
)
ect.run_info = {"run_completed": True}

# NB the templates need to be in microvolts, (nc, nwt, nc)
ect.data = dict(average=full_templates * 1e6, std=np.ones_like(full_templates))  # we may need to set the std here as well
analyzer.extensions['templates'] = ect

ecsp = ComputeSpikeAmplitudes(analyzer)
ecsp.set_params()
ecsp.run_info = {"run_completed": True}
ecsp.data['amplitudes'] = sorting.ssl.load_spike_sorting_object('spikes', attribute='amps')['amps']
analyzer.extensions['spike_amplitudes'] = ecsp

# %%
print(f'Compute noise levels {time.time() - tstart}')
analyzer.compute('noise_levels', n_jobs=3, random_slices_kwargs=dict(num_chunks_per_segment=7))
print(f'Compute quality metrics {time.time() - tstart}')
analyzer.compute('quality_metrics')
print(f'Compute template metrics {time.time() - tstart}')
analyzer.compute('template_metrics', include_multi_channel_metrics=True)

# %%
print(f'Unit Refine {time.time() - tstart}')
import pandas as pd
import spikeinterface.curation as sc
# needs  huggingface_hub, which is installed in the full spike interface
# pip install spikeinterface[full]
# Apply the noise/not-noise model
noise_neuron_labels = sc.auto_label_units(
    sorting_analyzer=analyzer,
    repo_id="AnoushkaJain3/noise_neural_classifier_lightweight",
    trust_model=True,
)

noise_units = noise_neuron_labels[noise_neuron_labels['prediction']=='noise']
analyzer_neural = analyzer.remove_units(noise_units.index)

# Apply the sua/mua model
sua_mua_labels = sc.auto_label_units(
    sorting_analyzer=analyzer_neural,
    repo_id="AnoushkaJain3/sua_mua_classifier_lightweight",
    trust_model=True,
)

all_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
from pathlib import Path

OUT_PATH = Path('/Users/olivier/Documents/scratch/brainhack')
all_labels.to_parquet(Path.home().joinpath(OUT_PATH.joinpath(f'{pid}_unit_refine.pqt')))

spikes, clusters, channels = ssl.load_spike_sorting()
df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))

analyzer.set_sorting_property('unit_refine', all_labels['prediction'].to_numpy().astype(str))
analyzer.set_sorting_property('ibl_bitwise_fail', df_clusters['bitwise_fail'].to_numpy())

analyzer.set_sorting_property('sirena', np.load(OUT_PATH.joinpath('sirena.npy')))


# %%
import spikeinterface.widgets as sw
print(f'Compute correlograms and unit locations {time.time() - tstart}')
analyzer.compute(['correlograms', 'unit_locations', 'template_similarity'])

print(f'Saving to disk {time.time() - tstart}')
analyzer.save_as(format='zarr', folder=OUT_PATH.joinpath(pid))


# %%
print(f'Launching GUI {time.time() - tstart}')

sw.plot_sorting_summary(
    sorting_analyzer=analyzer,
    backend='spikeinterface_gui',
    displayed_unit_properties=['unit_refine', 'ibl_bitwise_fail', 'sirena']
)
#
#
