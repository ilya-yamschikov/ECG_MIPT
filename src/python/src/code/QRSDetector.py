import numpy as np
import logging
import matplotlib.pyplot as plt

import src.code.WaveletProcessor as WP
import src.code.calculator as clc


class WaveletBasedQRSDetector(object):
    def __init__(self, y, sampling_fq):
        self._y = y
        self._sampling_fq = sampling_fq
        self._record_time = len(y) / sampling_fq
        self._wt = WP.get_WT(self._y, self._get_R_peak_scale())
        self._y, self._wt = self._clip_border_effects(self._y, self._wt)
        self._wt = clc.normalize(self._wt, 'median_abs')
        self._mm = WP.get_mm_array(self._wt)
        self._mm = WP.filter_mm_array(self._mm, self._wt)
        mm_abs_values = [np.abs(self._wt[mm_i]) for mm_i in self._mm]
        self._avg_mm = np.median(mm_abs_values)

    def _get_R_peak_scale(self):
        return int(0.008 * self._sampling_fq)

    def _get_T_wave_scale(self):
        return int(0.064 * self._sampling_fq)

    def _get_high_search_limit(self, mm, wt):
        diffs = []
        for i in range(len(mm)-1):
            diff = np.abs(wt[mm[i+1]] - wt[mm[i]])
            diffs.append(diff)
        diffs = sorted(diffs)
        return diffs[-5]

    def _clip_border_effects(self, y, wt):
        scale = self._get_R_peak_scale()
        clip = scale * 10 / 2 # half points size
        y = y[clip:-clip]
        wt = wt[clip:-clip]
        self._clip_size = clip
        return y, wt

    def _threshold_mm(self, mm, wt, threshold):
        peaks = []
        for i in range(1, len(mm)-1):
            if wt[mm[i]] > 0 >= wt[mm[i+1]]:
                diff_size = wt[mm[i]] - wt[mm[i+1]]
                if wt[mm[i-1]] < 0:
                    prev_diff_size = wt[mm[i]] - wt[mm[i-1]]
                    if prev_diff_size > threshold or diff_size > threshold:
                        peaks.append(mm[i])
                else:
                    if diff_size > threshold:
                        peaks.append(mm[i])
            if i > 1 and len(peaks) > 1:
                if peaks[-2] == mm[i-2]: # neighbour peaks disallowed
                    if wt[peaks[-2]] > wt[peaks[-1]]:
                        del peaks[-1]
                    else:
                        del peaks[-2]
        return peaks

    def _get_R_peaks(self, beta):
        threshold = self._avg_mm * beta
        mm = self._threshold_mm(self._mm, self._wt, threshold)
        return np.array(mm)

    def _is_final_solution(self, mm):
        if len(mm) == 0:
            logging.warn('Empty solution cannot be the right one')
            return False
        distances = []
        for i in range(1, len(mm)):
            distances.append(mm[i] - mm[i-1])
        avg_peak_distance = np.mean(distances)
        for i in range(1, len(mm)):
            dist = mm[i] - mm[i-1]
            if dist < 0.6 * avg_peak_distance:
                return False
            if dist > 1.5 * avg_peak_distance:
                return False
        return True

    def _adjust_peaks(self, peaks):
        adjusted_peaks = []
        for peak in peaks:
            while (0 < peak < len(self._y) - 1) and \
                  (not (self._y[peak-1] <= self._y[peak] >= self._y[peak+1])):
                if self._y[peak-1] > self._y[peak]:
                    peak -= 1
                elif self._y[peak+1] > self._y[peak]:
                    peak += 1
                else:
                    logging.error('Smth strange happened')
            adjusted_peaks.append(peak)
        return np.array(adjusted_peaks)

    def _get_value_to_optimize(self, mm):
        if len(mm) < 1:
            logging.error('Not enough mod maxes for value: %d', len(mm))
            return np.inf
        distances = []
        for i in range(1, len(mm)):
            distances.append((mm[i] - mm[i-1]) / float(self._sampling_fq)) # in seconds
        mean_dst_sec = np.mean(distances)
        if mean_dst_sec > (60. / 40.): # < 50 bps
            strange_pulse_penalty = (mean_dst_sec - (60./40.)) ** 2
        elif mean_dst_sec < (60. / 150.): # > 180 bps
            strange_pulse_penalty = (1 / mean_dst_sec) - (150. / 60)
        else:
            strange_pulse_penalty = 0.
        return np.std(distances) + strange_pulse_penalty

    def _find_minimum(self, function, value_function, interval, stop_condition):
        logging.info('Start minimum search on [%f, %f]', *interval)
        i = 1
        while i < 100 and (interval[1] - interval[0]) > 0.025:
            left_res = function(interval[0])
            left_val = value_function(left_res)
            right_res = function(interval[1])
            right_val = value_function(right_res)
            candidate1 = interval[0] + (interval[1] - interval[0]) / 4.
            candidate2 = interval[0] + (interval[1] - interval[0]) * 3. / 4.
            res1 = function(candidate1)
            # if stop_condition(res1):
            #     logging.info('Found minimum at 1st candidate: %f', candidate1)
            #     return res1
            res2 = function(candidate2)
            # if stop_condition(res2):
            #     logging.info('Found minimum at 2nd candidate: %f', candidate2)
            #     return res2
            value1 = value_function(res1)
            value2 = value_function(res2)
            if (value1 > left_val and value1 > right_val) or (value2 > left_val and value2 > right_val)\
                    or (left_val < value1 and value1 > value2 and value2 < right_val)\
                    or (left_val > value1 and value1 < value2 and value2 > right_val):
                logging.error('Minimum search error - non convex function: [%f, %f, %f, %f]', left_val, value1, value2, right_val)
            if value1 > value2:
                interval = [candidate1, interval[1]]
            else:
                interval = [interval[0], candidate2]
            logging.debug('Minimum search %d iteration is over - new interval [%f, %f]', i, *interval)
            i += 1
        res = res1 if value1 < value2 else res2
        if not stop_condition(res):
            logging.error('Solution doesn\'t satisfy stop condition')
        logging.info('Minimum search finished by %d iterations f(%f)=%f', i, candidate1, value1)
        return res

    def search_for_R_peaks(self):
        interval = [1., self._get_high_search_limit(self._mm, self._wt) / self._avg_mm]
        peaks = self._find_minimum(self._get_R_peaks, self._get_value_to_optimize, interval, self._is_final_solution)
        peaks = self._adjust_peaks(peaks)
        peaks = peaks + self._clip_size # shift peaks as they were clipped
        return peaks

    def _get_T_wave(self, R_peaks):
        pulse_fq = len(R_peaks) / self._record_time
        _smooth_y = clc.filter_to_range(self._y, self._sampling_fq, [pulse_fq / 1., pulse_fq * 15.])
        plt.plot(_smooth_y, 'g-')
        plt.show()

    def visualize_detector(self):
        __, p = plt.subplots(2, sharex=True)
        p[0].plot(self._y, 'r-')
        p[1].plot(self._wt, 'g-')
        p[1].plot(self._mm, self._wt[self._mm], 'r^')
        plt.show()