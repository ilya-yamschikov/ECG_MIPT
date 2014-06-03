import logging

from src.test import ECGDependentTest
from src.code.ECG_layout import *

logging.basicConfig(level=logging.DEBUG)

class LayoutTest(ECGDependentTest):
    def test_simple_layout(self):
        x = self._ecg.getTiming()
        y = self._ecg.getLowFreq()
        sampling_frequency = self._ecg.getDataFrequency()
        layout = generateSimpleLayout(y, sampling_frequency)
        drawLayout(x, y, layout)