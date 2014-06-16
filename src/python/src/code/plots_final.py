# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.interpolate import spline

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

def fft_comparison():
    ecg_healthy = loader.MouseECG(r'..\..\..\..\data\new_data\3_1.wav')
    ecg_ischemic = loader.MouseECG(r'..\..\..\..\data\new_data\3_3.wav')

    y_high_healthy = ecg_healthy.getHighFreq()
    y_high_ischemic = ecg_ischemic.getHighFreq()

    normalize=True
    if normalize:
        y_high_healthy = clc.normalize(y_high_healthy, 'energy=1', ecg_healthy.getDataFrequency())
        y_high_ischemic = clc.normalize(y_high_ischemic, 'energy=1', ecg_ischemic.getDataFrequency())

    def get_fft(y, fq):
        fft = np.abs(np.fft.rfft(y)) / (fq / 2.)
        f = fq / 2. * np.linspace(0.0, 1.0, len(y)/2 + 1)
        return f, fft

    f_healthy, fft_healthy = get_fft(y_high_healthy, ecg_healthy.getDataFrequency())
    f_ischemic, fft_ischemic = get_fft(y_high_ischemic, ecg_ischemic.getDataFrequency())

    # plt.title(u'Мышь 3')
    plt.plot(f_healthy, fft_healthy, 'g-', linewidth=1.0, label=u'До операции')
    plt.plot(f_ischemic, fft_ischemic, 'r-', linewidth=1.0, label=u'Развитие ишемии')
    plt.grid(True)
    plt.xlabel(u'Частота, Гц')
    plt.xlim([0., 2000.])
    plt.xticks(np.linspace(0., 2000., 9))
    # plt.vlines(200., 0., np.max(fft_ischemic), 'k', linewidth=2.5, zorder=10)
    plt.ylabel(u'Интенсивность (нормированная)')
    plt.legend()
    plt.savefig(r'..\..\logs\ft_comparison.png')

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

def draw_CV_on_window_center():
    x_ecg = np.array([-0.415263158, -0.336315789, -0.257368421, -0.178421053, -0.099473684, -0.035957895, -0.020526316, -0.015784211, 0.0, 0.036094737, 0.059157895, 0.069473684, 0.103842105, 0.188842105, 0.249368421, 0.373315789, 0.442473684, 0.468421053, 0.531842105, 0.576526316, 0.584736842, 0.648421053])
    y_ecg = np.array([-1414, -1738, -333, -1846, -981, 2585, 11990, 14476, 22367, 16746, 5720, -4333, -10386, -2819, -3143, -766, 3342, 3882, 1504, -1954, -2170, -2063])
    x_ecg_new = np.linspace(np.min(x_ecg), np.max(x_ecg), len(x_ecg) * 5)
    y_ecg_new = spline(x_ecg, y_ecg, x_ecg_new)

    x_CV = np.array(np.linspace(-0.475, 0.525, 41))
    # y_CV = [0.8811, 0.8757, 0.8649, 0.7838, 0.6595, 0.5297, 0.6054, 0.6270, 0.6378, 0.6270, 0.6486, 0.6378, 0.6378, 0.6216, 0.5892, 0.6324, 0.7027, 0.7243, 0.7351, 0.7351, 0.7081, 0.7135, 0.6270, 0.6270, 0.6486, 0.6486, 0.6378, 0.6216, 0.6000, 0.6270, 0.7892, 0.8270, 0.8378, 0.8054, 0.7405, 0.7297, 0.6541, 0.6811, 0.8378, 0.8595]
    y_CV = np.array([0.7135, 0.627, 0.627, 0.6486, 0.6486, 0.6378, 0.6216, 0.6, 0.627, 0.7892, 0.827, 0.8378, 0.8054, 0.7405, 0.7297, 0.6541, 0.6811, 0.8378, 0.8595, 0.8811, 0.8757, 0.8649, 0.7838, 0.6595, 0.5297, 0.6054, 0.627, 0.6378, 0.627, 0.6486, 0.6378, 0.6378, 0.6216, 0.5892, 0.6324, 0.7027, 0.7243, 0.7351, 0.7351, 0.7081, 0.7135])
    x_CV_new = np.linspace(np.min(x_CV), np.max(x_CV), len(x_CV) * 5)
    y_CV_new = spline(x_CV, y_CV, x_CV_new)

    fig, ax1 = plt.subplots()
    ax1.plot(x_ecg / 6, y_ecg / float(0.8 * np.max(y_ecg)), 'k-', linewidth=2.)
    ax1.set_xlabel(u'Время, сек')
    ax1.set_ylabel(u'Интенсивность', color='k')
    for tl in ax1.get_yticklabels():
        tl.set_color('k')

    ax2 = ax1.twinx()
    # ax2.plot(x_CV_new / 6, y_CV_new, 'r-', linewidth=2.)
    ax2.plot(x_CV / 6, y_CV, 'r-', linewidth=2.)
    ax2.plot(x_CV / 6, y_CV, 'r^')
    plt.ylim([0.1, 1.0])
    ax2.set_ylabel(u'Оценка 5х5-fold кросс-валидации', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.grid(True)
    # plt.show()
    plt.savefig(r'..\..\logs\CV_on_window_center.png')

def draw_CV_on_window_size():
    x = [0.3, 0.4, 0.5, 0.6, 0.8, 1.3, 1.5, 1.8]
    y = [0.873, 0.8784, 0.8649, 0.8486, 0.8486, 0.7892, 0.8, 0.7514]

    # plt.title(u'Зависимость от размера окна')
    plt.plot(x, y, 'b-', linewidth=2.)
    plt.plot(x, y, 'b^', linewidth=2.)
    plt.xlabel(u'Размер окна $\Delta w$')
    plt.ylabel(u'Оценка 5х5-fold кросс-валидации')
    plt.ylim([0.725, 0.925])
    plt.grid(True)
    # plt.show()
    plt.savefig(r'..\..\logs\CV_on_window_size.png')

def draw_CV_on_fq_divisions():
    x = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
    y = [0.8892, 0.8919, 0.8919, 0.8919, 0.8865, 0.8784, 0.873, 0.8703, 0.8622, 0.8568]

    # plt.title(u'Зависимость от размера окна')
    plt.plot(x, y, 'b-', linewidth=2.)
    plt.plot(x, y, 'b^', linewidth=2.)
    plt.xlabel(u'Количество разбиений по частоте')
    plt.ylabel(u'Оценка 5х5-fold кросс-валидации')
    plt.ylim([0.845, 0.9])
    plt.grid(True)
    # plt.show()
    plt.savefig(r'..\..\logs\CV_on_fq_divisions.png')

def draw_roc_whole_ft_curve():
    x_sd = [0.0000, 0.0101, 0.0202, 0.0303, 0.0404, 0.0505, 0.0606, 0.0707, 0.0808, 0.0909, 0.1010, 0.1111, 0.1212, 0.1313, 0.1414, 0.1515, 0.1616, 0.1717, 0.1818, 0.1919, 0.2020, 0.2121, 0.2222, 0.2323, 0.2424, 0.2525, 0.2626, 0.2727, 0.2828, 0.2929, 0.3030, 0.3131, 0.3232, 0.3333, 0.3434, 0.3535, 0.3636, 0.3737, 0.3838, 0.3939, 0.4040, 0.4141, 0.4242, 0.4343, 0.4444, 0.4545, 0.4646, 0.4747, 0.4848, 0.4949, 0.5051, 0.5152, 0.5253, 0.5354, 0.5455, 0.5556, 0.5657, 0.5758, 0.5859, 0.5960, 0.6061, 0.6162, 0.6263, 0.6364, 0.6465, 0.6566, 0.6667, 0.6768, 0.6869, 0.6970, 0.7071, 0.7172, 0.7273, 0.7374, 0.7475, 0.7576, 0.7677, 0.7778, 0.7879, 0.7980, 0.8081, 0.8182, 0.8283, 0.8384, 0.8485, 0.8586, 0.8687, 0.8788, 0.8889, 0.8990, 0.9091, 0.9192, 0.9293, 0.9394, 0.9495, 0.9596, 0.9697, 0.9798, 0.9899, 1.0000]
    y_sd = [0.000000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.650000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.800000, 0.866667, 0.866667, 0.866667, 0.866667, 0.866667, 0.866667, 0.866667, 0.866667, 0.866667, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 1.000000]

    # x_lsd = [0.0000, 0.0101, 0.0202, 0.0303, 0.0404, 0.0505, 0.0606, 0.0707, 0.0808, 0.0909, 0.1010, 0.1111, 0.1212, 0.1313, 0.1414, 0.1515, 0.1616, 0.1717, 0.1818, 0.1919, 0.2020, 0.2121, 0.2222, 0.2323, 0.2424, 0.2525, 0.2626, 0.2727, 0.2828, 0.2929, 0.3030, 0.3131, 0.3232, 0.3333, 0.3434, 0.3535, 0.3636, 0.3737, 0.3838, 0.3939, 0.4040, 0.4141, 0.4242, 0.4343, 0.4444, 0.4545, 0.4646, 0.4747, 0.4848, 0.4949, 0.5051, 0.5152, 0.5253, 0.5354, 0.5455, 0.5556, 0.5657, 0.5758, 0.5859, 0.5960, 0.6061, 0.6162, 0.6263, 0.6364, 0.6465, 0.6566, 0.6667, 0.6768, 0.6869, 0.6970, 0.7071, 0.7172, 0.7273, 0.7374, 0.7475, 0.7576, 0.7677, 0.7778, 0.7879, 0.7980, 0.8081, 0.8182, 0.8283, 0.8384, 0.8485, 0.8586, 0.8687, 0.8788, 0.8889, 0.8990, 0.9091, 0.9192, 0.9293, 0.9394, 0.9495, 0.9596, 0.9697, 0.9798, 0.9899, 1.0000]
    # y_lsd = [0.000000, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.533333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.733333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 0.933333, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]

    x_lsd = [0.0000, 0.0101, 0.0202, 0.0303, 0.0404, 0.0505, 0.0606, 0.0707, 0.0808, 0.0909, 0.1010, 0.1111, 0.1212, 0.1313, 0.1414, 0.1515, 0.1616, 0.1717, 0.1818, 0.1919, 0.2020, 0.2121, 0.2222, 0.2323, 0.2424, 0.2525, 0.2626, 0.2727, 0.2828, 0.2929, 0.3030, 0.3131, 0.3232, 0.3333, 0.3434, 0.3535, 0.3636, 0.3737, 0.3838, 0.3939, 0.4040, 0.4141, 0.4242, 0.4343, 0.4444, 0.4545, 0.4646, 0.4747, 0.4848, 0.4949, 0.5051, 0.5152, 0.5253, 0.5354, 0.5455, 0.5556, 0.5657, 0.5758, 0.5859, 0.5960, 0.6061, 0.6162, 0.6263, 0.6364, 0.6465, 0.6566, 0.6667, 0.6768, 0.6869, 0.6970, 0.7071, 0.7172, 0.7273, 0.7374, 0.7475, 0.7576, 0.7677, 0.7778, 0.7879, 0.7980, 0.8081, 0.8182, 0.8283, 0.8384, 0.8485, 0.8586, 0.8687, 0.8788, 0.8889, 0.8990, 0.9091, 0.9192, 0.9293, 0.9394, 0.9495, 0.9596, 0.9697, 0.9798, 0.9899, 1.0000]
    y_lsd = [0.000000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.550000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 0.750000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]


    plt.plot(x_sd,y_sd,'k-', label=u'Спектр по всему сигналу')
    # plt.plot(x_sd,y_sd,'k-')
    plt.plot(x_lsd,y_lsd,'r-', label=u'Спектр по окну с центром в R пике')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0., 1.], [0., 1.], '--', color=(0.6, 0.6, 0.6))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'..\..\logs\old_vs_new_roc.png')

# generate_ecg_signal_plot()
# general_signal_view()
# just_fft()
# fft_comparison()
# R_peak_detection()
# draw_wavelet_ricker()
# draw_CV_on_window_center()
# draw_CV_on_window_size()
# draw_CV_on_fq_divisions()
draw_roc_whole_ft_curve()