from __future__ import print_function

import multiprocessing
import time
import quantities as pq
import numpy as np
import spyica.ica as ica
import spyica.orica as orica

from spikeinterface import NumpySorting
from spikeinterface import sortingcomponents as sc
from spyica.tools import clean_sources, cluster_spike_amplitudes, detect_and_align, \
    reject_duplicate_spiketrains


def compute_ica(cut_traces, n_comp, t_init, ica_alg='ica', n_chunks=0,
                chunk_size=0, num_pass=1, block_size=800, verbose=True):
    if ica_alg == 'ica' or ica_alg == 'orica':
        if verbose and ica_alg == 'ica':
            print('Applying FastICA algorithm')
        elif verbose and ica_alg == 'orica':
            print('Applying offline ORICA')
    else:
        raise Exception("Only 'ica' and 'orica' are implemented")

    # TODO use random snippets (e.g. 20% of the data) / or spiky signals for fast ICA
    if ica_alg == 'ica':
        scut_ica, A_ica, W_ica = ica.instICA(cut_traces, n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size)
    else:
        scut_ica, A_ica, W_ica = orica.instICA(cut_traces, n_comp=n_comp,
                                               n_chunks=n_chunks, chunk_size=chunk_size,
                                               numpass=num_pass, block_size=block_size)
    if verbose:
        t_ica = time.time() - t_init
        if ica_alg == 'ica':
            print('FastICA completed in: ', t_ica)
        elif ica_alg == 'orica':
            print('ORICA completed in:', t_ica)

    return scut_ica, A_ica, W_ica


def clean_sources_ica(s_ica, A_ica, W_ica, kurt_thresh=1, skew_thresh=0.2, verbose=True):
    # clean sources based on skewness and correlation
    cleaned_sources_ica, source_idx = clean_sources(s_ica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_ica = A_ica[source_idx]
    cleaned_W_ica = W_ica[source_idx]

    if verbose:
        print('Number of cleaned sources: ', cleaned_sources_ica.shape[0])

    return cleaned_sources_ica, cleaned_A_ica, cleaned_W_ica, source_idx


def cluster(traces, fs, cleaned_sources_ica, num_frames, clustering='mog', spike_thresh=5,
            keep_all_clusters=False, features='amp', verbose=True):
    if verbose:
        print('Clustering Sources with: ', clustering)

    t_start = 0 * pq.s
    t_stop = num_frames / float(fs) * pq.s

    if clustering == 'kmeans' or clustering == 'mog':
        # detect spikes and align
        detected_spikes = detect_and_align(cleaned_sources_ica, fs, traces,
                                           t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
        spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
        spike_trains, amps, nclusters, keep, score = \
            cluster_spike_amplitudes(detected_spikes, metric='cal',
                                     alg=clustering, features=features, keep_all=keep_all_clusters)
        if verbose:
            print('Number of spike trains after clustering: ', len(spike_trains))
        sst, independent_spike_idx, dup = \
            reject_duplicate_spiketrains(spike_trains, sources=cleaned_sources_ica)
        if verbose:
            print('Number of spike trains after duplicate rejection: ', len(sst))
    else:
        raise Exception("Only 'mog' and 'kmeans' clustering methods are implemented")

    return sst, independent_spike_idx


def set_times_labels(sst, fs):
    times = np.array([], dtype=int)
    labels = np.array([])
    for i_s, st in enumerate(sst):
        times = np.concatenate((times, (st.times.magnitude * fs).astype(int)))
        labels = np.concatenate((labels, np.array([i_s + 1] * len(st.times))))

    return NumpySorting.from_times_labels(times.astype(int), labels, fs)


def mask_traces(recording, traces, fs, sample_window_ms=2,
                percent_spikes=None, max_num_spikes=None,
                balance_spikes_on_channel=False):
    """
    Find mask based on spike peaks

    Parameters
    ----------
    recording: si.RecordingExtractor
        The input recording extractor
    traces: np.array(channels, num_samples)
        Traces extracted from recording
    fs: float
        Sampling frequency
    sample_window_ms: float, int, list, or None
        If float or int, it's a symmetric window
        If list, it needs to have 2 elements. Asymmetric window
        If None, all traces are used
    percent_spikes: float
        Percentage of spikes selected
        If None, all spikes are used
    max_num_spikes: int
        Maximum number of spikes allowed
        If None, all spikes are used
    balance_spikes_on_channel: bool
        If true, the number of samples taken from each channel depends on the total number of spikes on the channel
        If false, random subsampling
    Returns
    -------

    """
    if sample_window_ms is None:
        return traces, None, None

    # set sample window
    if isinstance(sample_window_ms, float) or isinstance(sample_window_ms, int):
        sample_window_ms = [sample_window_ms, sample_window_ms]
    sample_window = [int(sample_window_ms[0] * fs), int(sample_window_ms[1] * fs)]
    num_channels = recording.get_num_channels()
    peaks = sc.detect_peaks(recording)

    # subsampling
    if percent_spikes is not None:
        if max_num_spikes is not None and percent_spikes * len(peaks['sample_ind']) > max_num_spikes:
            percent_spikes = max_num_spikes / len(peaks['sample_ind'])
        if balance_spikes_on_channel:
            final_idxs = []
            for chan in np.arange(num_channels):
                occurrences = list(peaks['channel_ind']).count(chan)
                num_samples = occurrences * percent_spikes
                idxs = np.where(peaks['channel_ind'] == chan)[0]
                idxs = np.random.choice(idxs, int(num_samples))
                final_idxs.extend(list(idxs))
            final_idxs = sorted(final_idxs)
            peaks_subsamp = peaks['sample_ind'][final_idxs]
            print(len(peaks_subsamp))
        else:
            num_samples = len(peaks['sample_ind']) * percent_spikes
            peaks_subsamp = np.random.choice(peaks['sample_ind'], int(num_samples))
            print(len(peaks_subsamp))
    else:
        peaks_subsamp = peaks['sample_ind']

    # find idxs
    selected_idxs = set()
    t_init = time.time()
    for peak_ind in peaks_subsamp:
        idxs_spike = np.arange(peak_ind - sample_window[0], peak_ind + sample_window[1])
        selected_idxs = selected_idxs.union(set(idxs_spike))

    t_end = time.time() - t_init

    selected_idxs = np.array(list(selected_idxs))
    selected_idxs = selected_idxs[selected_idxs > 1]
    selected_idxs = selected_idxs[selected_idxs < recording.get_num_samples(0) - 1]

    print(f"Sample number for ICA: {len(selected_idxs)} from {recording.get_num_samples(0)}\nElapsed time: {t_end}")

    cut_traces = traces[:, selected_idxs]

    return cut_traces, selected_idxs, peaks_subsamp


class Mask:
    sample_window = []

    def __init__(self, recording, traces, fs, sample_window_ms=2, percent_spikes=None,
                 max_num_spikes=None, balance_spikes_on_channel=False):
        self.recording = recording
        self.traces = traces
        self.fs = fs
        self.sample_window_ms = sample_window_ms
        self.percent_spikes = percent_spikes
        self.max_num_spikes = max_num_spikes
        self.balance_spikes_on_channel = balance_spikes_on_channel

    def run(self):
        import multiprocessing as mp
        import concurrent.futures as cf

        if self.sample_window_ms is None:
            return self.traces, None, None

        # set sample window
        if isinstance(self.sample_window_ms, float) or isinstance(self.sample_window_ms, int):
            self.sample_window_ms = [self.sample_window_ms, self.sample_window_ms]
        self.sample_window = [int(self.sample_window_ms[0] * self.fs), int(self.sample_window_ms[1] * self.fs)]
        num_channels = self.recording.get_num_channels()
        peaks = sc.detect_peaks(self.recording)

        # subsampling
        if self.percent_spikes is not None:
            if self.max_num_spikes is not None and self.percent_spikes * len(peaks['sample_ind']) > self.max_num_spikes:
                percent_spikes = self.max_num_spikes / len(peaks['sample_ind'])
            if self.balance_spikes_on_channel:
                final_idxs = []
                for chan in np.arange(num_channels):
                    occurrences = list(peaks['channel_ind']).count(chan)
                    num_samples = occurrences * self.percent_spikes
                    idxs = np.where(peaks['channel_ind'] == chan)[0]
                    idxs = np.random.choice(idxs, int(num_samples))
                    final_idxs.extend(list(idxs))
                final_idxs = sorted(final_idxs)
                peaks_subsamp = peaks['sample_ind'][final_idxs]
                print(len(peaks_subsamp))
            else:
                num_samples = len(peaks['sample_ind']) * self.percent_spikes
                peaks_subsamp = np.random.choice(peaks['sample_ind'], int(num_samples))
                print(len(peaks_subsamp))
        else:
            peaks_subsamp = peaks['sample_ind']

        t_init = time.time()
        manager = mp.Manager()
        selected_idxs = manager.list()
        n_jobs = mp.cpu_count()
        executor = cf.ProcessPoolExecutor(max_workers=n_jobs, initializer=init_f, initargs=(self.sample_window, ))
        executor.map(mask, peaks_subsamp)

        t_end = time.time() - t_init

        selected_idxs = np.array(list(selected_idxs))
        selected_idxs = selected_idxs[selected_idxs > 1]
        selected_idxs = selected_idxs[selected_idxs < self.recording.get_num_samples(0) - 1]

        print(f"Sample number for ICA: {len(selected_idxs)} from {self.recording.get_num_samples(0)}\nElapsed time: {t_end}")

        cut_traces = self.traces[:, selected_idxs]

        return cut_traces, selected_idxs, peaks_subsamp


global _sample_window


def init_f(sample_window):
    global _sample_window
    _sample_window = sample_window


def mask(peak_ind):
    global _sample_window
    idxs_spike = np.arange(peak_ind - _sample_window[0], peak_ind + _sample_window[1])
    selected_idxs = selected_idxs.union(set(idxs_spike))
