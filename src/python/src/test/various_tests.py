import pywt
import time
import logging
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from src.code.calculator import getMainFrequency, get_fft
from src.test import ECGDependentTest


def draw_cwt_as_img(y, cwt, scales):
    __, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(y, 'g-')
    axarr[1].imshow(cwt, aspect='auto')
    plt.yticks(range(0, len(scales)), [str(s) for s in scales])
    plt.show()


class VariousTests(ECGDependentTest):

    def _wave(self, width, height, center, signal_length):
        w = np.zeros(signal_length)
        s = max(center - width/2, 0)
        f = center
        w[s:f] = np.linspace(0., height, f-s)
        s = center
        f = min(center + width/2, signal_length)
        w[s:f] = np.linspace(height, 0., f-s)
        return w

    def _periodic_signal(self, points_count=10000, scale=50., wave_func=None):
        if wave_func is None:
            wave_func = self._wave
        sample_fq = int(points_count / scale)
        x = np.linspace(0., 1., points_count) * scale
        y = np.zeros(points_count)
        for cent in range(0, points_count, sample_fq/2):
            y += wave_func(20, 1., cent, points_count)
        return x, y, sample_fq

    def test_fft(self):
        x, y, sample_fq = self._periodic_signal()
        # plt.plot(x,y,'g-')
        # plt.show()
        print getMainFrequency(y, sample_fq)

    def _draw_cwt_as_plots(self, y, cwt, scales):
        f, axarr = plt.subplots(1 + len(scales), sharex=True)
        axarr[0].plot(y, 'g-')
        for i in xrange(0, len(scales)):
            axarr[i+1].plot(cwt[i, :], 'g-')
            axarr[i+1].set_ylabel(str(scales[i]))
        plt.show()

    def test_wavelet(self):
        MOUSE = False
        # scales = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        scales = np.array([4, 8, 12, 16, 20, 24, 32])
        # scales = np.array([48, 56, 64, 82, 100])
        if MOUSE:
            scales *= 44
            ecg = self.ecg_mouse()
        else:
            ecg = self.ecg()
        # x, y, sample_fq = self._periodic_signal()
        x, y, sample_fq = (ecg.getTiming(), ecg.getLowFreq(), ecg.getDataFrequency())
        # y = aline(y, sample_fq)
        tt = time.time()
        # cwt = sig.cwt(y, sig.ricker, np.arange(1, 100, 2))
        cwt = sig.cwt(y, sig.ricker, scales)
        logging.info('CWT took %.3f sec to calculate' % (time.time() - tt))
        # self._draw_cwt_as_img(y, cwt, scales)
        self._draw_cwt_as_plots(y, cwt, scales)

    def test_imshow(self):
        data = np.array([[0.1, 0.5, 0.1], [0.2, 0.4, 0.2]])
        plt.imshow(data)
        plt.show()

    def test_draw_wavelet_ricker(self):
        width = .5
        points = int(10*width)
        w = sig.ricker(points, width)
        # plt.plot(np.linspace(0., 1., points) * width, w, 'r-')
        plt.plot(w, 'r-')
        plt.show()

    def test_fft_wavelet_ricker(self):
        width = 1.
        points = int(10*width)
        w = sig.ricker(points, width)
        fft, f = get_fft(w, 1000)
        max_fq = np.argmax(fft)
        logging.info('Peak frequency = %f', f[max_fq])
        __, p = plt.subplots(2)
        p[0].plot(w, 'r-')
        p[1].plot(f, fft, 'g-')
        plt.show()

    def test_draw_wavelet_morlet(self):
        w = 6
        s = 0.7
        points = int(100*w)
        w = sig.morlet(points, w, s)
        plt.plot(np.linspace(0., 1., points) * s, w, 'r-')
        plt.show()

    def test_draw_wavelet_db(self):
        [phi, psi, x] = pywt.Wavelet('db2').wavefun(level=4)
        plt.plot(phi, 'r-')
        plt.show()

    def test_draw_wavelet(self):
        # x, y, sample_fq = (self.ecg().getTiming(), self.ecg().getLowFreq(), self.ecg().getDataFrequency())
        x, y, sample_fq = (self.ecg_mouse().getTiming(), self.ecg_mouse().getLowFreq(), self.ecg_mouse().getDataFrequency())
        plt.plot(x,y,'g-')
        plt.show()

    def test_ultra_high_fq(self):
        x_length = 70.
        sampling_fq = 200
        x = np.linspace(0., 1., sampling_fq * x_length) * x_length
        y = np.sin(99. * x * (2. * np.pi))
        # y = np.empty(len(x))
        # for i in range(len(x)):
        #     y[i] = 1. if (i % 4) < 2 else -1.
        fft, f = get_fft(y, sampling_fq)
        __, p = plt.subplots(2)
        p[0].plot(x, y, 'r-')
        p[1].plot(f, fft, 'g-')
        plt.show()