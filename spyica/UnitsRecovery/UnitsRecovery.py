import spikeinterface.comparison as sc
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
import numpy as np
from spikeinterface import extract_waveforms, aggregate_units
from ..SpyICASorter import SpyICASorter
from ..preProcessing import subtract_templates, lin_map
from ..tools import clean_correlated_sources


def _set_sorters_params(sorters_params_dict, we_params_dict):
    if 'sorters_params' not in sorters_params_dict.keys():
        sorters_params_dict['sorters_params'] = {}
    if 'engine' not in sorters_params_dict.keys():
        sorters_params_dict['engine'] = 'loop'
    if 'engine_kwargs' not in sorters_params_dict.keys():
        sorters_params_dict['engine_kwargs'] = {}
    if 'verbose' not in sorters_params_dict.keys():
        sorters_params_dict['verbose'] = False
    if 'with_output' not in sorters_params_dict.keys():
        sorters_params_dict['with_output'] = True
    if 'docker_images' not in sorters_params_dict.keys():
        sorters_params_dict['docker_images'] = {}

    if 'load_if_exists' not in we_params_dict.keys():
        we_params_dict['load_if_exists'] = False
    if 'precompute_template' not in we_params_dict.keys():
        we_params_dict['precompute_template'] = ('average', )
    if 'ms_before' not in we_params_dict.keys():
        we_params_dict['ms_before'] = 3.
    if 'ms_after' not in we_params_dict.keys():
        we_params_dict['ms_after'] = 4.
    if 'max_spikes_per_unit' not in we_params_dict.keys():
        we_params_dict['max_spikes_per_unit'] = 500
    if 'return_scaled' not in we_params_dict.keys():
        we_params_dict['return_scaled'] = True
    if'dtype' not in we_params_dict.keys():
        we_params_dict['dtype'] = None
    return sorters_params_dict, we_params_dict


class UnitsRecovery:

    def __init__(self, sorters: list, recordings, gt=None, sorters_params={}, we_params={}, well_detected_score=.7,
                 isi_thr=.3, fr_thr=None, sample_window_ms=2, percentage_spikes=None, balance_spikes=False,
                 detect_threshold=5, method='locally_exclusive', skew_thr=0.1, n_jobs=4, parallel=False, **job_kwargs):
        """
        Apply spike sorting algorithm two times to increase its accuracy. After the first sorting, well detected units
        are removed from the recording. ICA is run on the "new recording" to increase its SNR and then ease the detection
        of small units. Finally, the spike sorting algorithm is run again on the ica-filtered recording.

        Multiple sorting algorithms can be run at the same time, each one on its own recording. The recovery can be run
        in parallel or in a loop. The former option is suggested if the number of recordings or sortings is high.
        Parameters
        ----------
        sorters: list
            list of sorters name to be run.
        recordings: list or dict
            list or dict of RecordingExtractors. If dict, the keys are sorter names.
        gt: list
            list of ground truth SortingExtractors.
        sorters_params: dict
            dict with keys the parameters of spikeinterface.sorters.run_sorters().
            If a parameter is not set, its default values is used.
        we_params:
            dict with keys the parameters of spikeinterface.core.extract_waveforms().
            If a parameter is not set, its default values is used.
        well_detected_score: float
            agreement score to mark a unit as well detected. Used only if gt is provided.
        isi_thr: float
            If the ISI violation ratio of a unit is above the threshold, it will be discarded.
        fr_thr: list
            list with 2 values. If the firing rate of a unit is not in the provided interval,
            it will be discarded.
        sample_window_ms: list or int
            If list [ms_before, ms_after] of recording selected for each detected spike in subsampling for ICA.
        percentage_spikes: float
            percentage of detected spikes to be used in subsampling for ICA. If None, all spikes are used.
        balance_spikes: bool
            If true, same percentage of spikes is selected channel by channel. If None, spikes are picked randomly.
            Used only if percentage_spikes is not None
        detect_threshold: float
            MAD threshold for spike detection in subsampling for ICA.
        method: str
            How to detect peaks:
            * 'by_channel' : peak are detected in each channel independently. (default)
            * 'locally_exclusive' : locally given a radius the best peak only is taken but
              not neighboring channels.
        skew_thr: float
            Skewness threshold for ICA sources cleaning. If the skewness is lower than the threshold,
            it will be discarded.
        n_jobs: int
            Number of parallel processes
        parallel: bool
            If True, the recovery is run in parallel for each sorter. If False, the recovery is run in loop.
        job_kwargs: dict
            Parameters for parallel processing of RecordingExtractors.

        Returns
        --------
        UnitsRecovery object
        """
        self._recordings = recordings
        self._sorters = sorters
        self._gt = gt
        self._selected_units = {}
        if gt is not None:
            self._comparisons = {}
        self._recordings_backprojected = []
        self._aggregated_sortings = {}
        if fr_thr is None:
            fr_thr = [3.5, 19.5]

        self._sorters_params, self._we_params = _set_sorters_params(sorters_params, we_params)

        assert len(sorters) == len(recordings), "The number of sorters must equal the number of recordings"
        self._sortings_pre = ss.run_sorters(sorters, recordings, working_folder='sorting_pre',
                                            sorter_params=self._sorters_params['sorters_params'],
                                            mode_if_folder_exists='overwrite', engine=self._sorters_params['engine'],
                                            engine_kwargs=self._sorters_params['engine_kwargs'],
                                            verbose=self._sorters_params['verbose'],
                                            with_output=self._sorters_params['with_output'],
                                            docker_images=self._sorters_params['docker_images'])

        if parallel:
            raise NotImplementedError()
        else:
            self._do_recovery_loop(well_detected_score, isi_thr, fr_thr, sample_window_ms, percentage_spikes,
                                   balance_spikes, detect_threshold, method, skew_thr, n_jobs, **job_kwargs)

        self._sortings_post = ss.run_sorters(sorters, self._recordings_backprojected, working_folder='sorting_post',
                                             sorter_params=self._sorters_params['sorters_params'],
                                             mode_if_folder_exists='overwrite', engine=self._sorters_params['engine'],
                                             engine_kwargs=self._sorters_params['engine_kwargs'],
                                             verbose=self._sorters_params['verbose'],
                                             with_output=self._sorters_params['with_output'],
                                             docker_images=self._sorters_params['docker_images'])

    def _do_recovery_loop(self, well_detected_score, isi_thr, fr_thr, sample_window_ms, percentage_spikes,
                          balance_spikes, detect_threshold, method, skew_thr, n_jobs, **job_kwargs):
        for i, key in enumerate(self._sortings_pre.keys()):
            we = extract_waveforms(self._recordings[i], self._sortings_pre[key], folder='waveforms',
                                   load_if_exists=self._we_params['load_if_exists'],
                                   precompute_template=self._we_params['precompute_template'],
                                   ms_before=self._we_params['ms_before'], ms_after=self._we_params['ms_after'],
                                   max_spikes_per_unit=self._we_params['max_spikes_per_unit'],
                                   return_scaled=self._we_params['return_scaled'], dtype=self._we_params['dtype'],
                                   overwrite=True, **job_kwargs)
            if self._gt is not None:
                comparison = sc.compare_sorter_to_ground_truth(tested_sorting=self._sortings_pre[key],
                                                               gt_sorting=self._gt[i])
                self._comparisons[key] = comparison
                self._selected_units[key] = comparison.get_well_detected_units(well_detected_score)
            else:
                isi_violation = st.compute_isi_violations(we)[0]
                good_isi = np.argwhere(np.array(list(isi_violation.values())) < isi_thr)[:, 0]

                firing_rate = st.compute_firing_rate(we)
                good_fr_idx_up = np.argwhere(np.array(list(firing_rate.values())) < fr_thr[1])[:, 0]
                good_fr_idx_down = np.argwhere(np.array(list(firing_rate.values())) > fr_thr[0])[:, 0]

                self._selected_units[key] = [unit for unit in range(self._sortings_pre[key].get_num_units())
                                             if unit in good_fr_idx_up and unit in good_fr_idx_down and unit in good_isi]

            templates = we.get_all_templates()
            print(self._selected_units)
            print(self._sortings_pre[key].get_num_units())
            print(self._sortings_pre[key].get_unit_ids())
            templates_dict = {str(unit): templates[unit-1] for unit in self._selected_units[key]}

            recording_subtracted = subtract_templates(self._recordings[i], self._sortings_pre[key],
                                                      templates_dict, we.nbefore, self._selected_units[key])

            sorter = SpyICASorter(recording_subtracted)
            sorter.mask_traces(sample_window_ms=sample_window_ms, percent_spikes=percentage_spikes,
                               balance_spikes_on_channel=balance_spikes, detect_threshold=detect_threshold,
                               method=method, **job_kwargs)
            sorter.compute_ica(n_comp='all')
            cleaning_result = clean_correlated_sources(self._recordings[i], sorter.W_ica, skew_thresh=skew_thr, n_jobs=n_jobs,
                                                       chunk_size=self._recordings[i].get_num_samples(0) // n_jobs,
                                                       **job_kwargs)
            sorter.A_ica[cleaning_result[1]] = -sorter.A_ica[cleaning_result[1]]
            sorter.W_ica[cleaning_result[1]] = -sorter.W_ica[cleaning_result[1]]
            sorter.source_idx = cleaning_result[0]
            sorter.cleaned_A_ica = sorter.A_ica[cleaning_result[0]]
            sorter.cleaned_W_ica = sorter.W_ica[cleaning_result[0]]

            ica_recording = lin_map(recording_subtracted, sorter.cleaned_W_ica)
            self._recordings_backprojected.append(lin_map(ica_recording, sorter.cleaned_A_ica.T))

    def compare_performance(self):
        for i, key in enumerate(self._sortings_post.keys()):
            self._aggregated_sortings[key] = aggregate_units([self._sortings_post[key], list(self._sortings_pre.values())[i]])
            comparison_post = sc.compare_sorter_to_ground_truth(tested_sorting=self._aggregated_sortings[key],
                                                                gt_sorting=self._gt[i])
            print('Performance before recovery:')
            print(self._comparisons[key].print_performance())
            print('###')
            print('Performance after recovery:')
            print(comparison_post.print_performance())

    @property
    def sortings_pre(self):
        return self._sortings_pre

    @property
    def sortings_post(self):
        return self._sortings_post

    @property
    def recordings_backprojected(self):
        return self._recordings_backprojected

    @property
    def comparisons(self):
        return self._comparisons

    @property
    def aggregated_sortings(self):
        return self._aggregated_sortings


def units_recovery(*args, **kwargs):
    return UnitsRecovery(*args, **kwargs)


units_recovery.__doc__ = UnitsRecovery.__doc__
