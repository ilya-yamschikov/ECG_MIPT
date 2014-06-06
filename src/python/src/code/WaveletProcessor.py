import logging
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


MAXIMUM_SCOPE = 1
MM_SCALES = [2, 8, 32, 64]


def get_WT(y, scale):
    if not isinstance(scale, (list, tuple)):
        scale = [scale]
    wavelet = sig.ricker
    return sig.cwt(y, wavelet, scale)[0,:]

def is_modulus_maximum(w, i):
    if (i < 0) or (i >= len(w)):
        raise ValueError("i = %d, len(w) = %d" % (i, len(w)))

    w = np.abs(w)

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
    for i in range(len(w)):
        if is_modulus_maximum(w, i):
            mm.append(i)
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