import logging
import time
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


MAXIMUM_SCOPE = 1
MM_SCALES = [2, 8, 32, 64]


def get_WT(y, scale):
    single_scale = not isinstance(scale, (list, tuple, np.ndarray))
    if single_scale:
        scale = [scale]
    wavelet = sig.ricker
    tt = time.time()
    wt = sig.cwt(y, wavelet, scale)
    if single_scale:
        wt = wt[0,:]
    logging.debug('WT took %.3f sec to calculate on %s scale (signal length %d)', (time.time() - tt), str(scale), len(y))
    return wt

def fq_to_ricker_scale(fq, sampling_fq):
    scale = sampling_fq * .2 / fq
    if scale < .5:
        logging.error('Ricker wavelet corrupted: scale = %f', scale)
    return scale

def ricker_scale_to_fq(scale, sampling_fq):
    fq = sampling_fq * .2 / scale
    return fq

def fq_range_to_scales(fq_min, fq_max, sampling_fq, detalization=0.2):
    scales_low = np.round(fq_to_ricker_scale(fq_max, sampling_fq), decimals=1)
    scales_high = np.round(fq_to_ricker_scale(fq_min, sampling_fq), decimals=1)
    points = int(np.log(scales_high/scales_low) / np.log(1 + detalization))
    scales = np.arange(points + 1)
    scales = scales_low * ((1 + detalization) ** scales)
    scales = np.append(scales, scales_high)
    # scales = np.arange(scales_low, scales_high, detalization)
    scales = np.round(scales, decimals=1)
    scales = np.unique(scales)
    return scales

def is_modulus_maximum(w, i):
    if (i < 0) or (i >= len(w)):
        raise ValueError("i = %d, len(w) = %d" % (i, len(w)))

    if i == 0:
        left_side = None
    else:
        left_side = w[max(0, i-1-MAXIMUM_SCOPE):(i-1)]

    if i == len(w) - 1:
        right_side = None
    else:
        right_side = w[(i+1):min(i+1+MAXIMUM_SCOPE, len(w)-1)]

    if left_side is None:
        return np.all(np.less_equal(right_side, w[i]))
    if right_side is None:
        return np.all(np.less_equal(left_side, w[i]))

    return (np.all(np.less(left_side, w[i])) and np.all(np.less_equal(right_side, w[i]))) or (
            np.all(np.less_equal(left_side, w[i])) and np.all(np.less(right_side,  w[i])))

def get_mm_array(w):
    mm = []
    tt = time.time()
    abs_w = np.abs(w)
    for i in range(len(abs_w)):
        if is_modulus_maximum(abs_w, i):
            mm.append(i)
    logging.debug('MM took %.3f sec to calculate. Signal %d length -> %d MM', (time.time() - tt), len(w), len(mm))
    return np.array(mm)
#--------
# FILTERS
#--------
def filter_too_close_mm(mm_array):
    THRESHOLD = 10
    mm_array = mm_array.tolist()
    i = 1
    while i < len(mm_array):
        if np.abs(mm_array[i] - mm_array[i-1]) < THRESHOLD:
            del mm_array[i]
        else:
            i += 1
    return np.array(mm_array)

# Three MM same sign in line - too much
def filter_same_sign_mm(mm_array, wt):
    mm_array = mm_array.tolist()
    i = 1
    while i < len(mm_array) - 1:
        wt_i = (wt[mm_array[i-1]], wt[mm_array[i]], wt[mm_array[i+1]])
        if (np.sign(wt_i[0]) == np.sign(wt_i[1])) and (np.sign(wt_i[1]) == np.sign(wt_i[2])):
            wt_i = np.abs(wt_i)
            min_id = np.argmin(wt_i)
            del mm_array[i - 1 + min_id]
        else:
            i += 1
    return np.array(mm_array)

def filter_mm_array(mm_array, wt):
    mm_array = filter_too_close_mm(mm_array)
    mm_array = filter_same_sign_mm(mm_array, wt)
    return mm_array

def connect_different_scales_mm(mm_new, mm_prev):
    mm_new_upd = []
    for mm in mm_prev:
        new_mm_idx = np.argmin(np.abs(mm_new - mm))
        mm_new_upd.append(mm_new[new_mm_idx])
    return np.array(mm_new_upd)

def get_MM_lines(y):
    scales = sorted(MM_SCALES, reverse=True)
    mm = {}
    prev_mm = None
    for scale in scales:
        wt = get_WT(y, scale)
        new_mm = get_mm_array(wt)
        if prev_mm is not None:
            new_mm = connect_different_scales_mm(new_mm, prev_mm)
        else:
            new_mm = filter_mm_array(new_mm, wt)
        prev_mm = new_mm
        mm[scale] = new_mm
        logging.debug('Calculated modulus maximum for scale %d', scale)
    return mm

#-----------------------------
#getting R peaks from mm array
#-----------------------------
def get_R_peaks(mm, wt, threshold):
    peaks = []
    for i in range(1, len(mm)-1):
        local_mm = mm[i-1:i+1]
        if wt[local_mm[1]] > 0 > wt[local_mm[2]]:
            strength = wt[local_mm[1]] - wt[local_mm[2]]
            if strength > threshold:
                peaks.append(local_mm[1])
    return np.array(peaks)


# -------------
# visualisation
# -------------
def draw_MM_lines(mm, y, wt=None):
    lines_matrix = [mm[key] for key in sorted(mm.keys())]
    lines_matrix = np.array(lines_matrix)
    __, p = plt.subplots(2 if wt is None else 3, sharex=True)
    p[0].plot(y, 'g-')
    y_lines = np.array(range(lines_matrix.shape[0]))
    for i in range(lines_matrix.shape[1]):
        x = lines_matrix[:, i]
        p[1].plot(x, y_lines, 'g-')
    if wt is not None:
        p[2].plot(wt, 'b-')
    plt.show()

def draw_MM_strengths(mm, y, wt, diffs=False, wavelet=None):
    assert isinstance(mm, (list, np.ndarray))
    __, p = plt.subplots(2, sharex=True)
    p[0].plot(y, 'g-')
    p[1].plot(wt,'b-')
    if wavelet is not None:
        p[1].plot(wavelet, 'r-')
    for i in range(len(mm)-1):
        _x = mm[i]
        _y = wt[mm[i]]
        if diffs:
            txt = str(np.abs(wt[mm[i]] - wt[mm[i+1]]))
        else:
            txt = str(np.abs(wt[mm[i]]))
        p[1].text(_x,_y,txt)
    plt.show()