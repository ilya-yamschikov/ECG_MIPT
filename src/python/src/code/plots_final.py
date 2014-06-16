# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import src.code.ECG_loader as loader
import src.code.calculator as clc
from src.code.QRSDetector import WaveletBasedQRSDetector
import src.code.WaveletProcessor as WP

from matplotlib import rc
font = {'family': 'Verdana',
        'weight': 'normal',
        'size': 16}
rc('font', **font)


def clip(x):
    const=0.1
    return x[int(len(x)*const):int(len(x)*(1-const))]

def generate_ecg_signal_plot():
    ecg = loader.PTB_ECG(r'..\..\..\..\data\ptb_database_csv\s0002_re')

    x = ecg.getTiming()
    y = ecg.getLowFreq()
    y = clc.filterSignal(y, 1., ecg.getDataFrequency(), filterType='highpass')

    plt.plot(x, y, 'k-', linewidth=2.5)
    plt.xlim([0., 4.5])
    plt.xlabel(u'Время, сек')
    plt.ylabel(u'Интенсивность, $\mu V$')
    plt.grid(True)
    plt.savefig(r'..\..\logs\sample_ecg.png')
    # plt.show()

def general_signal_view():
    ecg = loader.MouseECG(r'..\..\..\..\data\new_data\1_1.wav')

    x = ecg.getTiming()
    y_low = ecg.getLowFreq()
    y_high = ecg.getHighFreq()

    hide_y_ticks=False
    to_clip=False

    if to_clip:
        const=0.1
        x=x[int(len(x)*const):int(len(x)*(1-const))]
        y_low=y_low[int(len(y_low)*const):int(len(y_low)*(1-const))]
        y_high=y_high[int(len(y_high)*const):int(len(y_high)*(1-const))]

    __, p = plt.subplots(2, sharex=True)
    p[0].plot(x,y_low,'r-', linewidth=2.0)
    p[1].plot(x,y_high,'g-', linewidth=1.2)
    if hide_y_ticks:
        p[0].yaxis.set_ticks_position('none')
        plt.setp(p[0].get_yticklabels(), visible=False)
        p[1].yaxis.set_ticks_position('none')
        plt.setp(p[1].get_yticklabels(), visible=False)
    plt.xlabel(u'Время, сек.')
    plt.xlim([0., 1.25])
    # plt.ylim([-9000., 7000.])
    plt.show()

def just_fft():
    ecg = loader.MouseECG(r'..\..\..\..\data\new_data\5_4.wav')

    x = ecg.getTiming()
    y_high = ecg.getHighFreq()
    fq = ecg.getDataFrequency()

    to_clip=False
    normalize=True
    if to_clip:
        x=clip(x)
        y_high=clip(y_high)
    if normalize:
        y_high = clc.normalize(y_high, 'energy=1', fq)

    fft = np.abs(np.fft.rfft(y_high)) / (fq / 2.)
    f = fq / 2. * np.linspace(0.0, 1.0, len(y_high)/2 + 1)

    plt.plot(f, fft, 'b-')
    plt.xlim([0., 2000.])
    plt.gca().xaxis.set_ticks(np.arange(0., 2000., 250.))
    plt.xlabel(u'Частота, Гц')
    plt.ylabel(u'Интенсивность')
    plt.show()

def R_peak_detection():
    ecg = loader.MouseECG(r'..\..\..\..\data\new_data\11_5.wav')

    fq = ecg.getDataFrequency()
    x = ecg.getTiming()
    y = ecg.getLowFreq()
    y = clc.normalize(y, 'energy=1', fq)
    detector = WaveletBasedQRSDetector(y, fq, ecg.PULSE_NORM)
    wt = WP.get_WT(y, detector._get_R_peak_scale())

    peaks = detector.search_for_R_peaks()

    hide_y_ticks=True

    __, p = plt.subplots(3, sharex=True)
    p[0].plot(x, y, 'r-')
    p[0].set_title(u'Сигнал ЭКГ')
    p[1].plot(x, wt, 'g-')
    p[1].plot(x[peaks], wt[peaks], 'b^', markeredgewidth=3.0)
    p[1].set_title(u'Вейвлет-преобразование с найденными пиками')
    p[2].plot(x, y, 'r-')
    p[2].plot(x[peaks], y[peaks], 'b^', markeredgewidth=3.0)
    p[2].set_title(u'R-пики оригинального сигнала')
    if hide_y_ticks:
        p[0].yaxis.set_ticks_position('none')
        plt.setp(p[0].get_yticklabels(), visible=False)
        p[1].yaxis.set_ticks_position('none')
        plt.setp(p[1].get_yticklabels(), visible=False)
        p[2].yaxis.set_ticks_position('none')
        plt.setp(p[2].get_yticklabels(), visible=False)
    plt.xlabel(u'Время, сек.')
    plt.xlim([0., 1.25])
    plt.show()

def draw_wavelet_ricker():
    width = 100.
    points = int(10*width)
    w = sig.ricker(points, width)
    plt.plot(w, 'k-', )
    plt.gca().yaxis.set_ticks_position('none')
    plt.show()

# generate_ecg_signal_plot()
# general_signal_view()
# just_fft()
# R_peak_detection()
draw_wavelet_ricker()
