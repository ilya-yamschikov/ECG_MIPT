import logging
import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.test import ECGDependentTest
import src.code.calculator as clc

logging.basicConfig(level=logging.DEBUG)


class AlineTests(ECGDependentTest):
    def _test_aline(self, x,y,fq):
        new_y = clc.aline(y, fq)
        plt.plot(x, y, 'g-', x, new_y, 'r-')
        plt.show()

    def test_aline_synthetic(self):
        self._test_aline(self._synthetic_x, self._synthetic_y, self._synthetic_sampling_fq)

    def test_aline_ptb(self):
        self._test_aline(self._ecg.getTiming(), self._ecg.getLowFreq(), self._ecg.getDataFrequency())

    def _test_main_fq(self, x, sampling_fq):
        x = clc.aline(x, sampling_fq)
        main_fq = clc.getMainFrequency(x, sampling_fq)
        plt.plot(np.linspace(0., 1., len(x)), x, 'g-')
        plt.plot(np.array([0., (sampling_fq / main_fq) / len(x)]), np.zeros(2), 'r-', linewidth=2.0)
        plt.show()

    def test_main_fq_ptb(self):
        self._test_main_fq(self._ecg.getLowFreq(), self._ecg.getDataFrequency())

    def test_filter_to_range(self):
        signal = self._ecg.getLowFreq()
        y = clc.filter_to_range(signal, self._ecg.getDataFrequency(), clc.REASONABLE_MAIN_FQ)
        plt.plot(y, 'r-', self._ecg.getLowFreq(), 'g-')
        plt.show()