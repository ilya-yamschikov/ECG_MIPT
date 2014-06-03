from src.test import ECGDependentTest
from src.code import WaveletProcessor as WP
import matplotlib.pyplot as plt
import numpy as np


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
        WP.draw_MM_lines(mm,y)