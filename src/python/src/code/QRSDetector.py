import numpy as np
import logging

import src.code.WaveletProcessor as WP
import src.code.calculator as clc


class WaveletBasedQRSDetector(object):
    def __init__(self, y, sampling_fq):
        self._y = y
        self._sampling_fq = sampling_fq
        self._wt = WP.get_WT(self._y, self._get_wavelet_scale())
        self._wt = clc.normalize(self._wt, 'median_abs')
        self._mm = WP.get_mm_array(self._wt)
        self._mm = WP.filter_mm_array(self._mm, self._wt)
        mm_abs_values = [np.abs(self._wt[mm_i]) for mm_i in self._mm]
        self._avg_mm = np.median(mm_abs_values)

    def _get_wavelet_scale(self):
        return 0.007 * self._sampling_fq

    def _get_high_search_limit(self, mm, wt):
        diffs = []
        for i in range(len(mm)-1):
            diff = np.abs(wt[mm[i+1]] - wt[mm[i]])
            diffs.append(diff)
        diffs = sorted(diffs)
        return diffs[-5]

    def _threshold_mm(self, mm, wt, threshold):
        peaks = []
        for i in range(1, len(mm)-1):
            if wt[mm[i]] > 0 > wt[mm[i+1]]:
                diff_size = wt[mm[i]] - wt[mm[i+1]]
                if wt[mm[i-1]] < 0:
                    prev_diff_size = wt[mm[i]] - wt[mm[i-1]]
                    if prev_diff_size > threshold or diff_size > threshold:
                        peaks.append(mm[i])
                else:
                    if diff_size > threshold:
                        peaks.append(mm[i])
        return peaks

    def get_R_peaks(self, beta):
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

    def _get_value_to_optimize(self, mm):
        distances = []
        for i in range(1, len(mm)):
            distances.append(mm[i] - mm[i-1])
        return np.std(distances)

    def _find_minimum(self, function, value_function, interval, stop_condition):
        logging.info('Start minimum search on [%f, %f]', *interval)
        i = 1
        while i < 100:
            left_res = function(interval[0])
            if stop_condition(left_res):
                return left_res
            left_val = value_function(left_res)
            right_res = function(interval[1])
            if stop_condition(right_res):
                return right_res
            right_val = value_function(right_res)
            candidate1 = interval[0] + (interval[1] - interval[0]) / 4.
            candidate2 = interval[0] + (interval[1] - interval[0]) * 3. / 4.
            res1 = function(candidate1)
            if stop_condition(res1):
                return res1
            res2 = function(candidate2)
            if stop_condition(res2):
                return res2
            value1 = value_function(res1)
            value2 = value_function(res2)
            if (value1 > left_val and value1 > right_res) or (value2 > left_res and value2 > right_res)\
                    or (left_val < value1 and value1 > value2 and value2 < right_val)\
                    or (left_val > value1 and value1 < value2 and value2 > right_val):
                logging.error('Minimum search error - non convex function')
            if value1 > value2:
                interval = [candidate1, interval[1]]
            else:
                interval = [interval[0], candidate2]
            logging.info('Minimum search %d iteration is over - new interval [%f, %f]', i, interval)
            i += 1

    def search_for_R_peaks(self):
        interval = [.25, self._get_high_search_limit(self._mm, self._wt) / self._avg_mm]
        peaks = self._find_minimum(self.get_R_peaks, self._get_value_to_optimize, interval, self._is_final_solution)
        return peaks