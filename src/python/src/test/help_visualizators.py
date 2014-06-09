import numpy as np

import matplotlib.pyplot as plt

from src.test import ECGDependentTest
import src.code.calculator as clc

class ECGVisualizator(ECGDependentTest):

    def test_plot_ecg(self):
        y_low = self.ecg().getLowFreq()
        y_high = self.ecg().getHighFreq()
        x = self.ecg().getTiming()
        __, p = plt.subplots(2, sharex=True)
        p[0].plot(x,y_low,'r-')
        p[1].plot(x,y_high,'g-')
        plt.show()

    def test_plot_fft(self):
        ecg = self.ecg()

        fq = ecg.getDataFrequency()
        x = ecg.getTiming()
        y_low = ecg.getLowFreq()
        fft_low, f_low = clc.get_fft(y_low, fq)
        y_high = ecg.getHighFreq()
        fft_high, f_high = clc.get_fft(y_high, fq)

        has_original = hasattr(ecg, 'getSignal')
        if has_original:
            y = ecg.getSignal()
            fft, f = clc.get_fft(y, fq)
            plots_count = 6
        else:
            plots_count = 4

        __, p = plt.subplots(plots_count)
        p[0].plot(x, y_low, 'r-')
        p[1].plot(f_low,fft_low,'r-')
        p[2].plot(x,y_high, 'g-')
        p[3].plot(f_high,fft_high,'g-')
        if has_original:
            p[4].plot(x, y, 'b-')
            p[5].plot(f, fft, 'b-')
        plt.show()