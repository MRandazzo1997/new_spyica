from __future__ import print_function

import time
import quantities as pq
import numpy as np
import spyica.ica as ica
import spyica.orica as orica

from spikeinterface import NumpySorting
from spikeinterface import sortingcomponents as sc
from spyica.tools import clean_sources, cluster_spike_amplitudes, detect_and_align, \
    reject_duplicate_spiketrains


def compute_ica(cut_traces, n_comp, ica_alg='ica', n_chunks=0,
                chunk_size=0, num_pass=1, block_size=800, verbose=True, max_iter=200):
    if ica_alg == 'ica' or ica_alg == 'orica':
        if verbose and ica_alg == 'ica':
            print('Applying FastICA algorithm')
        elif verbose and ica_alg == 'orica':
            print('Applying offline ORICA')
    else:
        raise Exception("Only 'ica' and 'orica' are implemented")

    # TODO use random snippets (e.g. 20% of the data) / or spiky signals for fast ICA
    if ica_alg == 'ica':
        scut_ica, A_ica, W_ica = ica.instICA(cut_traces, n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size, max_iter=max_iter)
    else:
        scut_ica, A_ica, W_ica = orica.instICA(cut_traces, n_comp=n_comp,
                                               n_chunks=n_chunks, chunk_size=chunk_size,
                                               numpass=num_pass, block_size=block_size)

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


def mask_traces(recording, fs, sample_window_ms=2,
                percent_spikes=None, max_num_spikes=None,
                balance_spikes_on_channel=False, use_lambda=True):
    """
    Find mask based on spike peaks

    Parameters
    ----------
    recording: si.RecordingExtractor
        The input recording extractor
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
    use_lambda
    Returns
    -------
    cut_traces: numpy array
        Array with traces

    """
    if sample_window_ms is None:
        return recording.get_traces().astype('int16').T, None, None

    # set sample window
    if isinstance(sample_window_ms, float) or isinstance(sample_window_ms, int):
        sample_window_ms = [sample_window_ms, sample_window_ms]
    sample_window = [int(sample_window_ms[0] * fs / 1000), int(sample_window_ms[1] * fs / 1000)]
    num_channels = recording.get_num_channels()
    peaks = sc.detect_peaks(recording, progress_bar=True)

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
        else:
            num_samples = len(peaks['sample_ind']) * percent_spikes
            peaks_subsamp = np.random.choice(peaks['sample_ind'], int(num_samples))
    else:
        peaks_subsamp = peaks['sample_ind']

    print(f"Number of detected spikes: {len(peaks['sample_ind'])}")
    print(f"Number of sampled spikes: {len(peaks_subsamp)}")

    # find idxs
    t_init = time.time()
    if use_lambda:
        idxs_spike_low = map(lambda peak: np.arange(peak - sample_window[0], peak + sample_window[1]), peaks_subsamp)
        selected_idxs = np.unique(list(idxs_spike_low))
    else:
        selected_idxs = np.array([], dtype=int)
        for peak_ind in peaks_subsamp:
            selected_idxs = np.concatenate((selected_idxs, np.arange(peak_ind - sample_window[0], peak_ind + sample_window[1])), dtype=int)
        selected_idxs = np.sort(np.unique(selected_idxs))
    t_end = time.time() - t_init

    selected_idxs = np.array(sorted(list(selected_idxs)))
    selected_idxs = selected_idxs[selected_idxs > 1]
    selected_idxs = selected_idxs[selected_idxs < recording.get_num_samples(0) - 1]

    cut_traces = None

    if percent_spikes is not None:
        for res in np.split(selected_idxs, np.where(np.diff(selected_idxs) != 1)[0]+1):
            traces = recording.get_traces(start_frame=res[0], end_frame=res[-1]).astype("int16")
            if cut_traces is None:
                cut_traces = traces
            else:
                cut_traces = np.vstack((cut_traces, traces))
        cut_traces = cut_traces.T
    else:
        cut_traces = recording.get_traces().astype('int16').T[:, selected_idxs]

    print(f"Sample number for ICA: {len(selected_idxs)} from {recording.get_num_samples(0)}\nElapsed time: {t_end}")

    # cut_traces = traces[:, selected_idxs]
    cut_traces = np.asarray(cut_traces)
    print(f"Shape: {cut_traces.shape}")

    return cut_traces, selected_idxs, peaks_subsamp
