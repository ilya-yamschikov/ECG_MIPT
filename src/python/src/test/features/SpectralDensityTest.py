import mock
import logging
import numpy as np
import matplotlib.pyplot as plt

import src.code.calculator as clc
from src.code.features.SpectralDensity import SpectralDensity
from src.test import ECGDependentTest

class SpectralDensityTest(ECGDependentTest):
    feature = SpectralDensity()

    # test that fft contribution calculated in proper way:
    # - no signal length dependency
    # - no sampling frequency dependency
    # - invariant to constant multiplier
    def test_synthetic(self):
        sampling_fq = 1000.
        length = 120.010 * 2
        x = np.linspace(0., 1., int(sampling_fq * length)) * length
        low_amp, middle_amp, high_amp = 5.0, 1.0, 6.0
        y = low_amp * np.sin(x * (2 * np.pi))
        y += middle_amp * np.sin(66 * x * (2 * np.pi))
        y += high_amp * np.sin(300 * x * (2 * np.pi))

        mockedECG = mock.Mock()
        mockedECG.getHighFreq = mock.Mock(return_value=y)
        mockedECG.getDataFrequency = mock.Mock(return_value=sampling_fq)
        mockedECG._f = None
        mockedECG._fft = None

        # plt.plot(x,y,'g-')
        # plt.show()
        # fft,f = clc.get_fft(y, sampling_fq)
        # plt.plot(f,fft,'r-')
        # plt.show()

        high = self.feature.run(mockedECG, begin=285., end=315., normalized=True)
        high_med = self.feature.run(mockedECG, begin=50., end=315., normalized=True)
        medium = self.feature.run(mockedECG, begin=50., end=70., normalized=True)
        low = self.feature.run(mockedECG, begin=0.5, end=1.5, normalized=True)
        logging.info('high/low amplitude: %f, by fft: %f', (high_amp/low_amp) ** 2, high/low)
        logging.info('high+med/low amplitude: %f, by fft: %f', ((high_amp+medium)/low_amp) ** 2, (high_med)/low)

    def test_real(self):
        ecg = self.ecg()
        self.feature.run(ecg, begin=200., end=500., normalized=True, use_original_signal=True)