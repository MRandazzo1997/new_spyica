import numpy as np
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from tqdm import tqdm

class SubtractTemplates(BasePreprocessor):

    def __init__(self, recording, sorting, templates_dict, n_before, good_units=None):
        # templates = np.array(templates, dtype='int32')

        BasePreprocessor.__init__(self, recording)
        if good_units is None:
            unit_ids = sorting.get_unit_ids()
        else:
            unit_ids = good_units
            
        for unit in templates_dict.keys():
            assert int(unit) in unit_ids

        for i, segment in enumerate(recording._recording_segments):
            all_spikes, all_labels = sorting.get_all_spike_trains()[i]
            new_segment = SubtractRecordingSegment(segment, all_spikes, all_labels, templates_dict, n_before, unit_ids)
            self.add_recording_segment(new_segment)
        self._kwargs = dict(recording=recording.to_dict(), sorting=sorting.to_dict(), templates_dict=templates_dict,
                            n_before=n_before, good_units=good_units)


class SubtractRecordingSegment(BasePreprocessorSegment):

    def __init__(self, segment, all_spikes, all_labels, templates_dict, n_before, unit_ids):
        BasePreprocessorSegment.__init__(self, segment)
        self.all_spikes = all_spikes
        self.all_labels = all_labels
        self.templates = templates_dict
        self.n_before = n_before
        self.unit_ids = unit_ids

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = traces.shape[0]
        
        mask = np.where((start_frame <= self.all_spikes) & (self.all_spikes < end_frame))[0]
        spikes_window = self.all_spikes[mask]
        labels_window = self.all_labels[mask]
        
        for i in tqdm(range(len(spikes_window)), ascii=True, desc="subtracting spikes"):
            if labels_window[i] in self.unit_ids:
                st = spikes_window[i]
                temp = np.array(self.templates[str(int(labels_window[i]))])
                start = int(st - self.n_before - start_frame)
                end = start + temp.shape[0]
                if start < 0:
                    tmp = temp[-start:]
                    traces[:end] -= tmp
                elif end > traces.shape[0]:
                    end = traces.shape[0] - 1
                    traces[start:end] -= temp[:end - start]
                else:
                    traces[start:end] -= temp
        # for unit_id in self.good_units:
        #     idxs = np.argwhere(st_labels == unit_id)[:, 0]
        #     unit_st = np.array(self.st)[idxs]
        #     templates = np.array(self.templates[str(unit_id)]).astype('int32')
        #     extr_idx = self.extr_idxs[str(unit_id)]
        #     max_id = np.abs(templates[:, extr_idx]).argmax()
        #
        #     for st in unit_st:
        #         if st in range(start_frame, end_frame):
        #             start = st - max_id - start_frame
        #             end = start + templates.shape[0]
        #             if start < 0:
        #                 tmp = templates[-start:]
        #                 traces[:end] -= tmp
        #             else:
        #                 if end > traces.shape[0]:
        #                     end = traces.shape[0] - 1
        #                     templates = templates[:end - start]
        #                 traces[start:end] -= templates

        traces = traces[:, channel_indices]
        return traces


def subtract_templates(*args):
    return SubtractTemplates(*args)


subtract_templates.__doc__ = SubtractTemplates.__doc__
