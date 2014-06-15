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

    def test_multibeat_time_n_freq(self):
        sampling_fq = 1000
        sample_length = 1.
        sample_x = np.linspace(0., 1., int(sampling_fq * sample_length)) * sample_length
        sample_y = np.cos(sample_x * 2 * np.pi)
        parts = [0.0, 0.4, 0.8, 1.0]
        fqs = [30., 40, 10.]
        amps = [2.0, 4., 2.0]

        x = np.zeros(1)
        y = np.array(sample_y[0])
        R_peaks = [0]
        samples_count = 5
        for s in range(samples_count):
            y_part = np.copy(sample_y)
            for i in range(len(parts)-1):
                part_begin = int(len(sample_x) * parts[i])
                part_end = int(len(sample_x) * parts[i+1]) + 1
                y_part[part_begin:part_end] += amps[i] * np.sin(fqs[i] * sample_x[part_begin:part_end] * 2 * np.pi)
            x = np.append(x, np.copy(sample_x[1:]) + x[-1])
            y = np.append(y, y_part[1:])
            R_peaks.append(R_peaks[-1] + len(y_part)-1)

        # missed beats
        # R_peaks = R_peaks[2:]

        # plt.plot(x, y, 'r-')
        # plt.plot(x[R_peaks], y[R_peaks], 'b^')
        # plt.show()
        # scales = WP.fq_range_to_scales(20., 50., sampling_fq)
        # wt = WP.get_WT(y, scales)
        # draw_cwt_as_img(y, wt, scales)

        y = clc.normalize(y, type='energy=1', sampling_fq=sampling_fq)

        # calculator = LocalizedSpectralDensity.WaveletBasedCalculator()
        calculator = LocalizedSpectralDensity.FourierBasedCalculator({'interval': [30., 200.], 'peak width': 0.008})
        # len_ranges = [[0, 0.4], [0.4, 0.8], [0.8, 1.0]]
        len_ranges = [[0, 1.0]]
        for len_range in len_ranges:
            energy = calculator.calc_energy(y, R_peaks, sampling_fq, 20., 50., len_range[0], len_range[1])
            logging.info('Energy in [%f, %f] slice: %f', len_range[0], len_range[1], energy)

    def test_ecg(self):
        logging.basicConfig(level=logging.INFO)

        ecg = self.ecg_mouse()
        feature = LocalizedSpectralDensity()
        # len_ranges = [[0, 0.3333], [0.3333, 0.6666], [0.6666, 1.0]]
        len_ranges = [[0.06, 0.56]]
        fq_begin, fq_end = 25, 50
        for len_range in len_ranges:
            energy = feature.run(ecg, len_range[0], len_range[1], fq_begin ,fq_end, calc_type='fft')
            logging.info('Energy in [%f, %f] slice on fq [%f, %f]: %f', len_range[0], len_range[1], fq_begin ,fq_end, energy)

    def test_fft_integrator(self):
        calculator = LocalizedSpectralDensity.FourierBasedCalculator({'interval': [30., 200.], 'peak width': 0.008})

        f = np.array([0.0, 0.5, 1.0])
        fft = np.array([1., 2., 1.])
        integral = calculator.integrate_fft(f, fft, 0.25, 0.75)
        self.assertAlmostEqual(integral, 0.875, places=3)

        f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        fft = np.array([1., 3., 2., 3., 3.])
        integral = calculator.integrate_fft(f, fft, 0.25, 1.75)
        self.assertAlmostEqual(integral, 3.875, places=3)

        f = np.array([3., 6., 9., 12.])
        fft = np.array([10., 12., 7., 11.])
        integral = calculator.integrate_fft(f, fft, 5., 11.)
        self.assertAlmostEqual(integral, 56.833, places=3)

        f = np.array([10., 11., 12., 13., 14., 15., 16.])
        fft = np.array([10., 11., 12., 13., 14., 15., 16.])
        integral = calculator.integrate_fft(f, fft, 12., 14.)
        self.assertAlmostEqual(integral, 26.0, places=1)