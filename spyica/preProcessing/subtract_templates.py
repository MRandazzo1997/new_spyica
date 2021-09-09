from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class SubtractTemplates(BasePreprocessor):

    def __init__(self, recording, templates_mask):

        BasePreprocessor.__init__(self, recording)

        for segment in recording._recording_segments:
            new_segment = SubtractRecordingSegment(segment, templates_mask)
            self.add_recording_segment(new_segment)
        self._kwargs = dict(recording=recording.to_dict(), templates_mask=templates_mask)


class SubtractRecordingSegment(BasePreprocessorSegment):

    def __init__(self, segment, templates_mask):
        BasePreprocessorSegment.__init__(self, segment)
        self.mask = templates_mask

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        traces_subt = traces - self.mask
        traces_subt = traces_subt[:, channel_indices]
        return traces_subt


def subtract_templates(recording, mask):
    return SubtractTemplates(recording, mask)


subtract_templates.__doc__ = SubtractTemplates.__doc__