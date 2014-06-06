from src.test import ECGDependentTest
from src.code import WaveletProcessor as WP
import matplotlib.pyplot as plt
import numpy as np
import src.code.calculator as clc
import scipy.signal as sig


class WPTest(ECGDependentTest):
    def test_ECG_mod_max(self):
        y = self.ecg().getLowFreq()
        x = np.array(range(len(y)))
        wt = WP.get_WT(y, 64)
        mm = WP.get_mm_array(wt)
        __, p = plt.subplots(2, sharex=True)
        p[0].plot(x, y, 'g-')
        p[1].plot(x, wt, 'r-')
        p[1].plot(mm, wt[mm], 'r^')
        plt.show()

    def test_ECG_full_MM(self):
        y = self.ecg().getLowFreq()
        mm = WP.get_MM_lines(y)
        wt = WP.get_WT(y, 64)
        WP.draw_MM_lines(mm,y,wt)

    def test_draw_strengths(self):
        scale = 12
        ecg = self.ecg()
        y = ecg.getLowFreq()
        wt = WP.get_WT(y, scale)
        wt = clc.normalize(wt, 'median_abs')
        mm = WP.get_mm_array(wt)
        mm = WP.filter_mm_array(mm, wt)
        WP.draw_MM_strengths(mm, y, wt, wavelet=100*sig.ricker(scale*10, scale))