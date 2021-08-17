import numpy as np
from spikeinterface import NumpySorting
from .linear_map import LinearMapFilter
from ..SpyICASorter.SpyICASorter import SpyICASorter


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ICAFilter(LinearMapFilter):

    def __init__(self, recording, sample_window_ms=1, percent_spikes=None, balance_spikes_on_channel=False,
                 max_num_spikes=None, clean=False):
        if isinstance(recording, NumpySorting):
            raise Exception("import a NumpySorting recording")
        sorter = SpyICASorter(recording)
        sorter.mask_traces(sample_window_ms=sample_window_ms, percent_spikes=percent_spikes,
                           balance_spikes_on_channel=balance_spikes_on_channel, max_num_spikes=max_num_spikes)
        sorter.compute_ica('all')
        self.A = sorter.A_ica
        source_idx = recording.ids_to_indices(recording.get_channel_ids())
        if clean:
            source_idx = []
            chan_loc = recording.get_channel_locations()
            num_channels = recording.get_num_channels()

            # find closest channels
            max_ids = np.argmax(sorter.A_ica, axis=1)
            dist = np.sqrt(np.square(chan_loc[:, 0] - chan_loc[:, 0, np.newaxis]) +
                           np.square(chan_loc[:, 1] - chan_loc[:, 1, np.newaxis]))
            closest = [[list(np.where(dist[:, i] < 60.0)[0])] for i in range(num_channels)]

            for chan in range(recording.get_num_channels()):
                max_chan = max_ids[chan]
                closest_val = sorter.A_ica[chan, closest[max_chan]]
                # if np.abs(np.sum(closest_val)) > max(np.max(closest_val), np.abs(np.min(closest_val))) * 0.66:
                if np.abs(np.sum(sorter.A_ica[chan])) > max(np.max(sorter.A_ica[chan]), np.abs(np.min(sorter.A_ica[chan]))):
                    source_idx.append(chan)
            print("cleaned: ", len(source_idx))

        LinearMapFilter.__init__(self, recording, sorter.W_ica, source_idx)
        self._kwargs = dict(recording=recording.to_dict(), sample_window_ms=sample_window_ms,
                            percent_spikes=percent_spikes, balance_spikes_on_channel=balance_spikes_on_channel,
                            max_num_spikes=max_num_spikes, clean=clean)


def ica_filter(recording, sample_window_ms=1, percent_spikes=None, balance_spikes_on_channel=False,
               max_num_spikes=None, clean=False):
    filt = ICAFilter(recording, sample_window_ms=sample_window_ms, percent_spikes=percent_spikes,
                     balance_spikes_on_channel=balance_spikes_on_channel, max_num_spikes=max_num_spikes,
                     clean=clean)
    return filt


ica_filter.__doc__ = ICAFilter.__doc__
