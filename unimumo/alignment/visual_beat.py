import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def d_x(im):
    d_im = np.zeros(im.shape)
    d_im[:, 0] = im[:, 0]
    for c in range(1, im.shape[1]):
        d_im[:, c] = im[:, c] - im[:, c - 1]
    new_im = -d_im
    new_im = np.clip(new_im, 0, None)
    vimpact = np.squeeze(np.mean(new_im, 0))

    cut_percentile = 99
    fx = np.fabs(vimpact)
    pv = np.percentile(fx, cut_percentile)
    pvlow = np.percentile(fx, cut_percentile - 1)
    normfactor = pv
    ptile = (vimpact > pv).astype(float)
    pntile = (vimpact < -pv).astype(float)
    pboth = ptile + pntile
    einds = np.flatnonzero(pboth)
    lastind = -2
    for j in range(len(einds)):
        if (einds[j] == (lastind + 1)):
            vimpact[einds[j]] = 0
        else:
            vimpact[einds[j]] = pvlow

    return vimpact / normfactor


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calc_directogram_and_kinematic_offset(skel):
    directogram = np.zeros((18, skel.shape[0] - 1))
    gg_count = 0
    for t in range(1, skel.shape[0]):
        skel_mov = np.zeros((skel.shape[1], 3))
        angs = []
        for k in range(skel.shape[1]):
            # if np.sum(np.abs(skel[t, k, :2])) < 1e-3 or np.sum(np.abs(skel[t - 1, k, :2])) < 1e-3:
            #     gg_count += 1
            #     continue
            # if skel[t, k, 2] < 0.1 or skel[t - 1, k, 2] < 0.1:
            #     continue
            skel_mov[k] = skel[t, k, :3] - skel[t - 1, k, :3]
            angs.append(angle_between(skel[t, k, :3], skel[t - 1, k, :3]))

        fx = skel_mov[:, 0]
        fy = skel_mov[:, 1]
        fz = skel_mov[:, 2]
        # ang = np.arctan2(fy, fx) + np.pi
        # ang = angle_between(skel[t, :, :3], skel[t-1, :, :3])
        # ang = np.arccos(skel[t, :, :3], skel[t-1, :, :3]) + np.pi
        amp = np.sqrt(fx * fx + fy * fy + fz * fz)
        # print(ang, ang.shape, amp.shape)
        angs = np.asarray(angs)
        # print(angs.shape, amp.shape)
        ahis1, cbinbounds1 = np.histogram(angs.ravel(), bins=18, range=(0, 2 * np.pi),
                                          weights=amp.ravel())

        directogram[:, t - 1] = ahis1
    vimpact = d_x(directogram)
    return directogram, vimpact


class Node(object):
    def __init__(self, frame,  offset_weight, sampling_rate, local_auto_correlation, prev_node=None):
        self.frame = frame
        self.offset_weight = offset_weight
        self.sampling_rate = sampling_rate
        self.local_auto_correlation = local_auto_correlation
        self.prev_node = prev_node
        self.cum_score = None


def getVisualTempogram(onset_envelope, window_length, sampling_rate):
    win_length = int(round(window_length * sampling_rate))
    sr = sampling_rate
    hop_length = 1
    center = True
    window = 'hann'
    norm = np.inf
    ac_window = librosa.filters.get_window(window, win_length, fftbins=True)
    # Center the autocorrelation windows
    n = len(onset_envelope)
    if center:
        onset_envelope = np.pad(onset_envelope, int(win_length // 2),
                                mode='linear_ramp', end_values=[0, 0])
    # Carve onset envelope into frames
    odf_frame = librosa.util.frame(onset_envelope,
                                   frame_length=win_length,
                                   hop_length=hop_length)
    # Truncate to the length of the original signal
    if center:
        odf_frame = odf_frame[:, :n]

    odf_frame = librosa.util.normalize(odf_frame, axis=0, norm=norm)
    norm_columns = True

    if norm_columns:
        # Window, autocorrelate, and normalize
        result = librosa.util.normalize(
            librosa.autocorrelate(odf_frame * ac_window[:, np.newaxis], axis=0), norm=norm, axis=0)
    else:
        result = librosa.autocorrelate(odf_frame * ac_window[:, np.newaxis], axis=0)
        result = np.true_divide(result, np.max(result.ravel()))

    tempo_bpms = librosa.tempo_frequencies(result.shape[0], hop_length=hop_length, sr=sr)
    return tempo_bpms, result


def get_candid_peaks(vimpact, sampling_rate):
    single_frame = 1.0 / sampling_rate
    delta = 0.015
    time_params = dict(
        pre_max_time=2.0 * single_frame,
        post_max_time=2.0 * single_frame,
        pre_avg_time=5.0 * single_frame,
        post_avg_time=5.0 * single_frame,
        wait_time=2.0 * single_frame,
        delta=0.015,
    )
    tp_keys = time_params.keys()
    for p in tp_keys:
        time_params[p] = int(round(sampling_rate * time_params[p]))

    dparams = dict(
        pre_max=time_params['pre_max_time'],
        post_max=time_params['post_max_time'],
        pre_avg=time_params['pre_avg_time'],
        post_avg=time_params['post_avg_time'],
        wait=time_params['wait_time'],
        delta=delta
    )
    peakinds = librosa.util.peak_pick(x=vimpact, **dparams)
    peakvals = vimpact[peakinds]
    return peakinds, peakvals


def weight_unary_objective(kin_offset_value, unary_weight=None):
    if unary_weight is None:
        unary_weight = 1.0

    return unary_weight * kin_offset_value


def autocor_binary_objective(left, right, local_auto_correlation, sampling_rate, binary_weight=None):
    if binary_weight is None:
        binary_weight = 1.0
    period = int(np.abs(left - right))
    if period < len(local_auto_correlation):
        score = (local_auto_correlation[period] - 1)
    else:
        score = -1
    if period / sampling_rate < 0.25 or period / sampling_rate > 3.75:
        score = -1
    return binary_weight * score


def window_func(left, right, fps, max_separation=4):
    return np.fabs(left - right) < int(max_separation * fps)


def find_optimal_paths(candid_visual_beats, auto_correlation, sampling_rate):
    """
    :param candid_visual_beats: list of tuple - (frame, kin_offset_value)
    :param auto_correlation: tempogram calculated from visual beats
    :param sampling_rate: video fps
    :return:
    """

    nodes = []

    for i, beat in enumerate(candid_visual_beats):
        nodes.append(Node(beat[0],  beat[1], sampling_rate, auto_correlation[:, i]))

    nodes[0].prev_node = None
    nodes[0].cum_score = weight_unary_objective(nodes[0].offset_weight)
    current_segment = []
    segments = []
    for n in range(1, len(nodes)):
        current_node = nodes[n]
        current_segment.append(current_node)
        options = []
        j = n - 1
        while j >= 0 and window_func(current_node.frame, nodes[j].frame, sampling_rate):
            options.append(nodes[j])
            j = j - 1
        if len(options) == 0:
            current_node.prev_node = None
            current_node.cum_score = weight_unary_objective(current_node.offset_weight)
            segments.append(current_segment)
            current_segment = []
        else:
            best_choice = options[0]
            # may change 0.02
            const = 0.05
            best_score = options[0].cum_score + const * autocor_binary_objective(current_node.frame, best_choice.frame,
                                                                         auto_correlation[:, n - 1], sampling_rate)
            for o in range(1, len(options)):
                score = options[o].cum_score + const * autocor_binary_objective(current_node.frame, options[o].frame,
                                                                        auto_correlation[:, n - 1 - o], sampling_rate)
                if score > best_score:
                    best_choice = options[o]
                    best_score = score
            current_node.prev_node = best_choice
            current_node.cum_score = best_score + weight_unary_objective(current_node.offset_weight)

    if len(current_segment) > 0:
        segments.append(current_segment)
    sequences = []
    for S in segments:
        seq = []
        max_node = S[0]
        max_score = max_node.cum_score
        for n in range(len(S)):
            if S[n].cum_score > max_score:
                max_node = S[n]
                max_score = max_node.cum_score
        trace_node = max_node
        while trace_node.prev_node is not None:
            seq.append((trace_node.frame, trace_node.offset_weight, trace_node.sampling_rate,
                        trace_node.local_auto_correlation))
            trace_node = trace_node.prev_node
        seq.reverse()
        sequences.append(seq)
    return sequences