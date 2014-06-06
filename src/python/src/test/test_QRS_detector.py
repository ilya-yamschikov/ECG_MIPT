import matplotlib.pyplot as plt

from src.test import ECGDependentTest
from src.code.QRSDetector import WaveletBasedQRSDetector
import src.code.WaveletProcessor as WP


class QRSDetectorTest(ECGDependentTest):
    def test_QRS_detector_given_threshold(self):
        beta = 44.
        y = self.ecg().getLowFreq()
        detector = WaveletBasedQRSDetector(y, self.ecg().getDataFrequency())
        wt = WP.get_WT(y, detector._get_R_peak_scale())
        peaks = detector._get_R_peaks(beta)
        self._draw_QRS(peaks, wt, y)

    def test_R_peaks(self):
        ecg = self.ecg()
        y = ecg.getLowFreq()
        detector = WaveletBasedQRSDetector(y, ecg.getDataFrequency())
        # detector.visualize_detector()
        wt = WP.get_WT(y, detector._get_R_peak_scale())
        peaks = detector.search_for_R_peaks()
        self._draw_QRS(peaks, wt, y)

    def test_T_wave(self):
        y = self.ecg().getLowFreq()
        detector = WaveletBasedQRSDetector(y, self.ecg().getDataFrequency())
        peaks = detector.search_for_R_peaks()
        detector._get_T_wave(peaks)

    def _draw_QRS(self, peaks, wt, y):
        __, p = plt.subplots(2, sharex=True)
        p[0].plot(y, 'g-')
        if len(peaks) > 0:
            p[0].plot(peaks, y[peaks], 'r^')
        p[1].plot(wt, 'b-')
        if len(peaks) > 0:
            p[1].plot(peaks, wt[peaks], 'r^')
        plt.show()