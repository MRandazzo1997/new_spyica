import numpy as np
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ..tools import clean_sources
import warnings as w


class LinearMapFilter(BasePreprocessor):
    name = 'Filter'

    def __init__(self, recording, matrix, clean=True, kurt_thresh=1, skew_thresh=0.2):
        self.recording = recording
        if isinstance(matrix, np.ndarray):
            self.M = [matrix]
        else:
            self.M = matrix
        self._clean = clean
        if isinstance(kurt_thresh, float) or isinstance(kurt_thresh, int):
            kurt_thresh = [kurt_thresh]
        if isinstance(skew_thresh, float) or isinstance(skew_thresh, int):
            skew_thresh = [skew_thresh]
        if not recording.get_num_channels() == self.M[0].shape[0]:
            raise ArithmeticError(
                f"Matrix first dimension must be equal to number of channels: {recording.get_num_channels()}"
                f"It is: {self.M.shape[0]}")

        BasePreprocessor.__init__(self, recording)
        for i, parent_segment in enumerate(recording._recording_segments):
            segment = FilterRecordingSegment(parent_segment, self.M[i], skew_thresh[i], kurt_thresh[i], self._clean)
            self.add_recording_segment(segment)

        self._kwargs = dict(recording=recording.to_dict())

    def get_num_channels(self, segment_index):
        return len(self.get_channel_ids(segment_index))

    def get_channel_ids(self, segment_index):
        if self._recording_segments[segment_index].idxs is None:
            if self._clean: w.warn("Linear filter hasn't been applied yet")
            return self._main_ids
        else:
            return self._main_ids[self._recording_segments[segment_index].idxs]

    def get_channel_locations(self, segment_index, channel_ids=None, locations_2d=True):
        chan_locations = self.recording.get_channel_locations()
        chan_locations = chan_locations[self._recording_segments[segment_index].idxs]
        return chan_locations


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, M, skew_thresh=0.1, kurt_thresh=1, clean=True):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.M = M
        self._kurt_thresh = kurt_thresh
        self._skew_thresh = skew_thresh
        self._clean = clean
        self.idxs = None

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None)).T
        filtered_traces = self.M @ traces
        filtered_traces = filtered_traces[channel_indices, :]
        if self._clean:
            cleaned_traces, self.idxs = clean_sources(filtered_traces, kurt_thresh=self._kurt_thresh,
                                                      skew_thresh=self._skew_thresh)
        else:
            cleaned_traces = filtered_traces
        return cleaned_traces


def lin_filter(*args):
    return LinearMapFilter(*args)


lin_filter.__doc__ = LinearMapFilter.__doc__
