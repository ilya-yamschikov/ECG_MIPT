# -*- coding: utf-8 -*-

import numpy as np
import logging

import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family': 'Verdana',
        'weight': 'normal',
        'size': 22}
rc('font', **font)

from src.test import ECGDependentTest
import src.code.calculator as clc
from src.code.QRSDetector import WaveletBasedQRSDetector

class ECGVisualizator(ECGDependentTest):

    def test_plot_ecg(self):
        ecg = self.ecg_mouse()

        y_low = ecg.getLowFreq()
        y_high = ecg.getHighFreq()
        x = ecg.getTiming()

        to_clip=False
        if to_clip:
            const=0.1
            x=x[int(len(x)*const):int(len(x)*(1-const))]
            y_low=y_low[int(len(y_low)*const):int(len(y_low)*(1-const))]
            y_high=y_high[int(len(y_high)*const):int(len(y_high)*(1-const))]

        logging.info('General statistic: RMS_low=%f, RMS_high=%f', clc.RMS(y_low), clc.RMS(y_high))

        # __, p = plt.subplots(2, sharex=True)
        # p[0].plot(x,y_low,'r-', linewidth=2.0)
        # p[0].yaxis.set_ticks_position('none')
        # plt.setp(p[0].get_yticklabels(), visible=False)
        # p[1].plot(x,y_high,'g-', linewidth=1.2)
        # p[1].yaxis.set_ticks_position('none')
        # plt.setp(p[1].get_yticklabels(), visible=False)
        # p[1].set_title(u'200-2000 Гц')
        # plt.xlabel(u'Время, сек.')
        # plt.xlim([0., 1.25])
        # plt.ylim([-9000., 7000.])
        # plt.show()

        plt.plot(x,y_low,'r-', linewidth=2.0)
        plt.xlim([0., 1.2])
        plt.xlabel(u'Время, сек.', horizontalalignment='right')
        plt.gca().axes.yaxis.set_visible(False)
        # plt.ylim([-9000., 7000.])
        plt.show()

    def test_plot_fft(self):
        ecg = self.ecg_mouse()

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

    def test_plot_beat_fractions(self):
        ecg = self.ecg_mouse()
        x = ecg.getTiming()
        y = ecg.getLowFreq()

        R_detector = WaveletBasedQRSDetector(y, ecg.getDataFrequency(), ecg.PULSE_NORM)
        R_peaks = R_detector.search_for_R_peaks()

        # Mouse T wave:  clip = [0.2, 0.65]
        # Mouse QRS wave:  clip = [-0.15, 0.15]
        clip = [-0.25, 0.25]
        clip_left = R_peaks[:-1] + np.array(clip[0] * (R_peaks[1:] - R_peaks[:-1]), dtype=np.int)
        clip_right = R_peaks[:-1] + np.array(clip[1] * (R_peaks[1:] - R_peaks[:-1]), dtype=np.int)

        plt.plot(x, y, 'b-')
        plt.plot(x[R_peaks], y[R_peaks], 'r^')
        plt.plot(x[clip_left], y[clip_left], 'g^')
        plt.plot(x[clip_right], y[clip_right], 'g^')
        plt.show()