import numpy as np
import logging
import matplotlib.pyplot as plt

import src.code.calculator as clc
import src.code.WaveletProcessor as WP
from src.code.features.LocalizedSpectralDensity import LocalizedSpectralDensity
from src.test import ECGDependentTest
from src.test.various_tests import draw_cwt_as_img


class Test(ECGDependentTest):
    def test_synthetic_freq(self):
        sampling_fq = 1000
        signal_length = 150.
        x = np.array(np.linspace(0., 1., int(sampling_fq * signal_length))) * signal_length
        y = np.sin(1. * x * (2*np.pi))
        y += 10.1 * np.sin(30. * x * (2*np.pi))
        y += 4.5 * np.sin(110. * x * (2*np.pi))
        y = np.array(y)
        R_peaks = np.array([0, len(y) - 1])

        # plt.plot(x, y, 'g-')
        # plt.show()
        # fft, f = clc.get_fft(y, sampling_fq)
        # plt.plot(f, fft, 'r-')
        # plt.show()

        # fq_ranges = [[15, 45], [90, 130]]
        fq_ranges = [[15, 130]]

        feature = LocalizedSpectralDensity()
        for fq_range in fq_ranges:
            # scales = WP.fq_range_to_scales(fq_range[0], fq_range[1], sampling_fq)
            # wt = WP.get_WT(y, scales)
            # energy = feature._integrate_wt(R_peaks, wt, sampling_fq, fq_range[1] - fq_range[0])
            # fqs_array = [WP.ricker_scale_to_fq(scale, sampling_fq) for scale in scales]
            # draw_cwt_as_img(y, wt, fqs_array)
            energy = feature._calc_energy(y, R_peaks, sampling_fq, fq_range[0], fq_range[1])
            logging.info('Energy in [%f, %f] Hz: %f', fq_range[0], fq_range[1], energy)

    def test_synthetic_time(self):
        sampling_fq = 1000
        signal_length = 60.
        x = np.array(np.linspace(0., 1., int(sampling_fq * signal_length))) * signal_length
        y = np.zeros(len(x))
        x1 = x[:len(x)/3]
        x2 = x[len(x)/3:2*len(x)/3]
        x3 = x[2*len(x)/3:]
        y[:len(x)/3] = 0.1 * np.sin(10. * x1 * (2*np.pi))
        y[len(x)/3:2*len(x)/3] = 0.5 * np.sin(10. * x2 * (2*np.pi))
        y[2*len(x)/3:] = 4.5 * np.sin(10. * x3 * (2*np.pi))
        R_peaks = np.array([0, len(y) - 1])

        # plt.plot(x, y, 'g-')
        # plt.show()
        # fft, f = clc.get_fft(y, sampling_fq)
        # plt.plot(f, fft, 'r-')
        # plt.show()

        feature = LocalizedSpectralDensity()
        len_ranges = [[0, 0.3333], [0.3333, 0.6666], [0.6666, 1.0]]
        for len_range in len_ranges:
            energy = feature._calc_energy(y, R_peaks, sampling_fq, 3., 20., len_range[0], len_range[1])
            logging.info('Energy in [%f, %f] slice: %f', len_range[0], len_range[1], energy)

    def test_ecg_ptb(self):
        logging.basicConfig(level=logging.INFO)

        ecg = self.ecg()
        feature = LocalizedSpectralDensity()
        len_ranges = [[0, 0.3333], [0.3333, 0.6666], [0.6666, 1.0]]
        fq_begin, fq_end = 200, 400
        for len_range in len_ranges:
            energy = feature.run(ecg, len_range[0], len_range[1], fq_begin ,fq_end)
            logging.info('Energy in [%f, %f] slice on fq [%f, %f]: %f', len_range[0], len_range[1], fq_begin ,fq_end, energy)