import numpy as np
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ..SpyICASorter import SpyICASorter
from spyica.tools import clean_sources
from typing import List
from spikeinterface.core.baserecording import BaseRecordingSegment


class ICAOnRecording(SpyICASorter, BasePreprocessor):

    def __init__(self, recording, sample_window_ms, kurt_thresh=1, skew_thresh=0.2, clean=True, **random_chunk_kwargs):
        BasePreprocessor.__init__(self, recording)
        SpyICASorter.__init__(self, recording)
        self._recording_segments: List[BaseRecordingSegment] = []
        self.mask_traces(sample_window_ms=sample_window_ms, percent_spikes=None,
                         balance_spikes_on_channel=True)
        self.compute_ica(n_comp='all')
        for recording_segment in recording._recording_segments:
            rec_segment = ICAOnRecordingSegment(recording_segment, self.W_ica, self.A_ica, clean=clean,
                                                kurt_thres=kurt_thresh, skew_thresh=skew_thresh)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict())
        self._kwargs.update(random_chunk_kwargs)


class ICAOnRecordingSegment(BasePreprocessorSegment):

    def __init__(self, recording_segment, W, A, clean=True, kurt_thres=1, skew_thresh=0.2):
        BasePreprocessorSegment.__init__(self, recording_segment)
        self.W = W
        self.A = A
        self.clean = clean
        self.kurt_thresh = kurt_thres
        self.skew_thresh = skew_thresh
        self.source_idx = None

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        ica_traces = np.matmul(self.W, traces.T)
        ica_traces = ica_traces[:, channel_indices]
        if self.clean:
            cleaned_sources_ica, self.source_idx = clean_sources(ica_traces, kurt_thresh=self.kurt_thresh,
                                                                 skew_thresh=self.skew_thresh)
        else:
            cleaned_sources_ica = ica_traces
        return cleaned_sources_ica


def ica_on_recording(recording, sample_window_ms, kurt_thresh=1, skew_thresh=0.2):
    return ICAOnRecording(recording, sample_window_ms, kurt_thresh=kurt_thresh,
                          skew_thresh=skew_thresh, clean=True)


ica_on_recording.__doc__ = ICAOnRecording.__doc__
