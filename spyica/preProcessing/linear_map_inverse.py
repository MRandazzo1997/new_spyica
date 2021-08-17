import numpy as np
import scipy.stats as ss
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class LinearMapInverse(BasePreprocessor):
    name = 'Backprojection'

    def __init__(self, recording, matrix, mean):
        self.recording = recording
        if isinstance(matrix, list):
            self.matrix = np.array(matrix)
        else:
            self.matrix = matrix
        self.mean = mean
        BasePreprocessor.__init__(self, recording)

        for seg in recording._recording_segments:
            segment = InverseRecordingSegment(seg, matrix, mean)
            self.add_recording_segment(segment)

        self._kwargs = dict(recording=recording.to_dict(), matrix=matrix, mean=mean)


class InverseRecordingSegment(BasePreprocessorSegment):

    def __init__(self, segment, matrix, mean):
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        self.matrix = matrix
        if isinstance(mean, list):
            mean = np.array(mean)
        self.mean = mean
        BasePreprocessorSegment.__init__(self, segment)

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None)).T
        # mixing matrix must be transposed when passed as argument
        if isinstance(traces, list):
            traces = np.array(traces)
        backproj_traces = (self.matrix @ traces) + self.mean[:, np.newaxis]
        backproj_traces = backproj_traces[channel_indices, :]
        sk_sp = ss.skew(backproj_traces, axis=1)
        # invert sources with positive skewness
        backproj_traces[sk_sp > 0] = -backproj_traces[sk_sp > 0]
        return backproj_traces.T


def backprojection(*args):
    return LinearMapInverse(*args)


backprojection.__doc__ = LinearMapInverse.__doc__
