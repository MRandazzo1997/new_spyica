from spikeinterface import NumpySorting
from .preProcessing import LinearMapFilter
from ..SpyICASorter.SpyICASorter import SpyICASorter


class ICAFilter(SpyICASorter, LinearMapFilter):

    def __init__(self, recording, sample_window_ms=1, percent_spikes=None, balance_spikes_on_channel=False,
                 max_num_spikes=None):
        if isinstance(recording, NumpySorting):
            raise Exception("import a NumpySorting recording")
        SpyICASorter.__init__(self, recording)
        self.mask_traces(sample_window_ms=sample_window_ms, percent_spikes=percent_spikes,
                         balance_spikes_on_channel=balance_spikes_on_channel, max_num_spikes=max_num_spikes)
        self.compute_ica('all')
        LinearMapFilter.__init__(self, recording, self.W_ica)
