import matplotlib.pyplot as plt
import numpy as np
import logging

from src.test import ECGDependentTest
from src.code.QRSDetector import WaveletBasedQRSDetector
import src.code.WaveletProcessor as WP


class QRSDetectorTest(ECGDependentTest):
    def test_QRS_detector_given_threshold(self):
        beta = 20.
        ecg = self.ecg_mouse()

        y = ecg.getLowFreq()
        detector = WaveletBasedQRSDetector(y, ecg.getDataFrequency(), ecg.PULSE_NORM)
        wt = WP.get_WT(y, detector._get_R_peak_scale())
        peaks = detector._get_R_peaks(beta)
        logging.info('Error value: %f', detector._get_value_to_optimize(peaks))
        y, wt = detector._clip_border_effects(y, wt)
        self._draw_QRS(peaks, wt, y)

    def test_R_peaks(self):
        ecg = self.ecg_mouse()
        x = ecg.getTiming()
        y = ecg.getLowFreq()
        detector = WaveletBasedQRSDetector(y, ecg.getDataFrequency(), ecg.PULSE_NORM)
        # detector.visualize_detector()
        wt = WP.get_WT(y, detector._get_R_peak_scale())
        peaks = detector.search_for_R_peaks()
        self._draw_QRS(peaks, wt, y, x=x)

    def test_optimization_curve(self):
        ecg = self.ecg_mouse()
        y = ecg.getLowFreq()
        detector = WaveletBasedQRSDetector(y, ecg.getDataFrequency(), ecg.PULSE_NORM)
        search_interval = [0.1, detector._get_high_search_limit(detector._mm, detector._wt) / detector._avg_mm]
        search_grid = np.linspace(search_interval[0], search_interval[1], 240)
        curve = np.zeros(len(search_grid))
        for i in range(len(search_grid)):
            mm = detector._get_R_peaks(search_grid[i])
            curve[i] = detector._get_value_to_optimize(mm)
        logging.info('Curve: [%s]', ', '.join([str(x) for x in curve]))
        plt.plot(search_grid, curve, 'r-')
        plt.show()


    def test_T_wave(self):
        ecg = self.ecg()
        y = ecg.getLowFreq()
        detector = WaveletBasedQRSDetector(y, ecg.getDataFrequency(), ecg.PULSE_NORM)
        peaks = detector.search_for_R_peaks()
        detector._get_T_wave(peaks)

    def _draw_QRS(self, peaks, wt, y, x=None):
        if x is None:
            x = np.arange(len(y))

        __, p = plt.subplots(2, sharex=True)
        p[0].plot(x, y, 'g-')
        if len(peaks) > 0:
            p[0].plot(x[peaks], y[peaks], 'r^')
        p[1].plot(x, wt, 'b-')
        if len(peaks) > 0:
            p[1].plot(x[peaks], wt[peaks], 'r^')
        plt.show()