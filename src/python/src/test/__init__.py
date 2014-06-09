import unittest
import numpy as np

from src.code.ECG_loader import PTB_ECG, MouseECG

PTB_FILE = r'..\..\..\..\data\ptb_database_csv\s0557_re'
MOUSE_FILE = r'..\..\..\..\data\new_data\18_1.wav'


class ECGDependentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        points = 2000
        scale = 20.
        cls._synthetic_sampling_fq = points / scale
        cls._synthetic_x = np.array(np.linspace(0., 1., points) * scale)
        cls._synthetic_y = np.sin(cls._synthetic_x * 20) + np.sin(cls._synthetic_x / 2.) + 1.

        cls._ecg = None
        cls._ecg_mouse = None

    def ecg(self):
        if self._ecg is None:
            self._ecg = PTB_ECG(PTB_FILE)
        return self._ecg

    def ecg_mouse(self):
        if self._ecg_mouse is None:
            self._ecg_mouse = MouseECG(MOUSE_FILE)
        return self._ecg_mouse