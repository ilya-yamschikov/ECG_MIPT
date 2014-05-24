import logging
import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.code.ECG_loader import PTB_ECG
from src.code.calculator import aline, getMainFrequency

logging.basicConfig(level=logging.DEBUG)

PTB_FILE = r'..\..\..\..\data\ptb_database_csv\s0001_re'


class AlineTests(unittest.TestCase):
    def _test_aline(self, x,y,fq):
        new_y = aline(y, fq)
        plt.plot(x, y, 'g-', x, new_y, 'r-')
        plt.show()

    def test_aline_synthetic(self):
        POINTS = 2000
        SCALE = 20.
        sampling_fq = POINTS / SCALE
        x = np.array(np.linspace(0., 1., POINTS) * SCALE)
        y = np.sin(x * 20) + np.sin(x / 2.) + 1.
        self._test_aline(x, y, sampling_fq)

    def test_aline_ptb(self):
        ecg = PTB_ECG(PTB_FILE)
        self._test_aline(ecg.getTiming(), ecg.getLowFreq(), ecg.getDataFrequency())

    def _test_main_fq(self, x, sampling_fq):
        x = aline(x, sampling_fq)
        main_fq = getMainFrequency(x, sampling_fq)
        plt.plot(np.linspace(0., 1., len(x)), x, 'g-')
        plt.plot(np.array([0., (sampling_fq / main_fq) / len(x)]), np.zeros(2), 'r-', linewidth=2.0)
        plt.show()

    def test_main_fq_ptb(self):
        ecg = PTB_ECG(PTB_FILE)
        self._test_main_fq(ecg.getLowFreq(), ecg.getDataFrequency())
