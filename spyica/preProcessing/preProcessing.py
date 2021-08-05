import numpy as np
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class LinearMapFilter(BasePreprocessor):
    name = 'Filter'

    def __init__(self, recording, matrix, ids):
        self.recording = recording
        if isinstance(matrix, list):
            self.M = np.asarray(matrix)
        else:
            self.M = matrix
        # if not recording.get_num_channels() == self.M.shape[0]:
        #     raise ArithmeticError(
        #         f"Matrix first dimension must be equal to number of channels: {recording.get_num_channels()}"
        #         f"It is: {self.M.shape[0]}")

        self.ids = ids

        BasePreprocessor.__init__(self, recording)
        for i, parent_segment in enumerate(recording._recording_segments):
            segment = FilterRecordingSegment(parent_segment, self.M, ids)
            self.add_recording_segment(segment)

        self._kwargs = dict(recording=recording.to_dict(), matrix=matrix, ids=ids)

    # def get_num_channels(self):
    #     return len(self.ids)
    #
    # def get_channel_ids(self):
    #     return np.asarray([str(i + 1) for i in self.ids], dtype='<U64')


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, M, ids):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.M = M
        self.ids = ids

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame).T
        filtered_traces = np.matmul(self.M, traces)
        # filtered_traces = filtered_traces[self.ids]
        if channel_indices is not None and channel_indices in self.ids:
            filtered_traces = filtered_traces[channel_indices, :]
        print(filtered_traces.shape, self.M.shape, traces.shape)
        return filtered_traces.T


def lin_filter(*args):
    return LinearMapFilter(*args)


lin_filter.__doc__ = LinearMapFilter.__doc__
