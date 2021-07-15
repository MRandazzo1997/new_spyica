import numpy as np
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.toolkit.utils import get_random_data_chunks
from ..SpyICASorter import mask_traces, compute_ica


class ICAOnRecording(BasePreprocessor):

    def __init__(self, recording, sample_window_ms):
        BasePreprocessor.__init__(self, recording)
        cut_rec, idxs, peaks = mask_traces(self._parent_recording, self.get_sampling_frequency(),
                                           sample_window_ms=sample_window_ms, percent_spikes=0.1,
                                           balance_spikes_on_channel=True)
        s_ica, W_ica, A_ica = compute_ica(cut_rec, n_comp='all')
        for recording_segment in recording._recording_segments:
            rec_segment = ICAOnRecordingSegment(recording_segment, W_ica, A_ica)
            self.add_recording_segment(rec_segment)


class ICAOnRecordingSegment(BasePreprocessorSegment):

    def __init__(self, recording_segment, W, A):
        BasePreprocessorSegment.__init__(self, recording_segment)
        self.W = W
        self.A = A

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        ica_traces = np.matmul(self.W, traces)
        ica_traces = ica_traces[:, channel_indices]
        return ica_traces


def ica_on_recording(recording, sample_window_ms):
    return ICAOnRecording(recording, sample_window_ms)


ica_on_recording.__doc__ = ICAOnRecording.__doc__
