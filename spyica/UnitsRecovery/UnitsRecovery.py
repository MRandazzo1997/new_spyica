import spikeinterface.comparison as sc
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
import numpy as np
from spikeinterface import extract_waveforms, aggregate_units
from ..SpyICASorter import SpyICASorter
from ..preProcessing import subtract_templates, lin_map
from ..tools import clean_correlated_sources


class UnitsRecovery:
    def __init__(self, sorters, recordings, gt=None, well_detected_score=.7, isi_thr=.3, fr_thr=None,
                 sample_window_ms=2, percentage_spikes=None, balance_spikes=False, detect_threshold=5,
                 method='locally_exclusive', progress_bar=True, skew_thr=0.1, n_jobs=4,
                 print_performances=False, **job_kwargs):
        self.recordings = recordings
        self.sorters = sorters
        self.selected_units = {}
        if gt is not None:
            self.comparisons = {}
        self.recordings_backprojected = []
        self.aggregated_sortings = {}
        if fr_thr is None:
            fr_thr = [3.5, 19.5]

        assert len(sorters) == len(recordings), "The number of sorters must equal the number of recordings"
        self.sortings_pre = ss.run_sorters(sorters, recordings, working_folder='sorting_pre',
                                           mode_if_folder_exists='overwrite')

        for i, key in enumerate(self.sortings_pre.keys()):
            we = extract_waveforms(recordings[i], self.sortings_pre[key], folder='waveforms',
                                   overwrite=True, progress_bar=progress_bar)
            if gt is not None:
                comparison = sc.compare_sorter_to_ground_truth(tested_sorting=self.sortings_pre[key],
                                                               gt_sorting=gt[i])
                self.comparisons[key] = comparison
                self.selected_units[key] = comparison.get_well_detected_units(well_detected_score)
            else:
                isi_violation = st.compute_isi_violations(we)[0]
                good_isi = np.argwhere(np.array(list(isi_violation.values())) < isi_thr)[:, 0]

                firing_rate = st.compute_firing_rate(we)
                good_fr_idx_up = np.argwhere(np.array(list(firing_rate.values())) < fr_thr[1])[:, 0]
                good_fr_idx_down = np.argwhere(np.array(list(firing_rate.values())) > fr_thr[0])[:, 0]

                self.selected_units[key] = [unit for unit in range(self.sortings_pre[key].get_num_units())
                                            if unit in good_fr_idx_up and unit in good_fr_idx_down and unit in good_isi]

            templates = we.get_all_templates()
            templates_dict = {str(unit): templates[unit] for unit in self.selected_units[key]}

            recording_subtracted = subtract_templates(recordings[i], self.sortings_pre[key],
                                                      templates_dict, we.nbefore, self.selected_units[key])

            sorter = SpyICASorter(recording_subtracted)
            sorter.mask_traces(sample_window_ms=sample_window_ms, percent_spikes=percentage_spikes,
                               balance_spikes_on_channel=balance_spikes, detect_threshold=detect_threshold,
                               method=method, **job_kwargs)
            sorter.compute_ica(n_comp='all')
            cleaning_result = clean_correlated_sources(recordings[i], sorter.W_ica, skew_thresh=skew_thr, n_jobs=n_jobs,
                                                       chunk_size=recordings[i].get_num_samples(0)//n_jobs,
                                                       progress_bar=progress_bar)
            sorter.A_ica[cleaning_result[1]] = -sorter.A_ica[cleaning_result[1]]
            sorter.W_ica[cleaning_result[1]] = -sorter.W_ica[cleaning_result[1]]
            sorter.source_idx = cleaning_result[0]
            sorter.cleaned_A_ica = sorter.A_ica[cleaning_result[0]]
            sorter.cleaned_W_ica = sorter.W_ica[cleaning_result[0]]

            ica_recording = lin_map(recording_subtracted, sorter.cleaned_W_ica)
            self.recordings_backprojected.append(lin_map(ica_recording, sorter.cleaned_A_ica.T))
        self.sortings_post = ss.run_sorters(sorters, self.recordings_backprojected, working_folder='sorting_post')

        for i, key in self.sortings_post.keys():
            self.aggregated_sortings[key] = aggregate_units([self.sortings_post[key], list(self.sortings_pre.values())[i]])
            if print_performances:
                comparison_post = sc.compare_sorter_to_ground_truth(tested_sorting=self.aggregated_sortings[key],
                                                                    gt_sorting=gt[i])
                print(f'Performance before recovery:\n{self.comparisons[i].print_performance()}')
                print('###')
                print(f'Performance after recovery:\n{comparison_post.print_performance()}')


def units_recovery(*kargs, **kwargs):
    return UnitsRecovery(*kargs, **kwargs)


units_recovery.__doc__ = UnitsRecovery.__doc__



