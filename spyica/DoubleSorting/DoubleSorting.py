from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class MatchedRecording(BasePreprocessor):

    def __init__(self, recording, matching_dict, good_units=None):
        self.recording = recording
        if good_units is None:
            good_units = self.sorting.get_units_ids()
        self.good_units = good_units
        self.matching_dict = matching_dict
        self.num_channels = self.recording.get_num_channels()
        self.num_samples = self.recording.get_num_samples(0)

        BasePreprocessor.__init__(self, self.recording)
        for recording_segment in self.recording._recording_segments:
            recording_segment = MatchedRecordingSegment(recording_segment, self.matching_dict, self.good_units,
                                                        self.num_samples, self.num_channels)
            self.add_recording_segment(recording_segment)

        self._kwargs = dict(recording=recording.to_dict(), matching_dict=matching_dict, good_units=good_units)


class MatchedRecordingSegment(BasePreprocessorSegment):

    def __init__(self, recording_segment, matching_dict, good_units, num_samples, num_channels):
        BasePreprocessorSegment.__init__(self, recording_segment)
        self.matching_dict = matching_dict
        self.good_units = good_units
        self.num_channels = num_channels
        self.num_samples = num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        match_traces = self.parent_recording_segment.get_traces()
        for unit in self.good_units:
            unit_starts = self.matching_dict[unit]
            for chan in range(self.num_channels):
                chan_starts = unit_starts[chan]
                tmp_channel = unit_starts[str(chan) + '_tmp']
                for st in range(len(chan_starts)):
                    st_start = chan_starts[st]
                    if st_start + len(tmp_channel) > self.num_samples:
                        match_traces[st_start:, chan] -= \
                            tmp_channel[:self.num_samples - st_start]
                    else:
                        match_traces[st_start:st_start + len(tmp_channel), chan] -= tmp_channel
        match_traces = match_traces[start_frame:end_frame, channel_indices]
        return match_traces


def match_recording(recording, matching_dict, good_units=None):
    mr = MatchedRecording(recording, matching_dict, good_units=good_units)
    return mr


match_recording.__doc__ = MatchedRecording.__doc__
