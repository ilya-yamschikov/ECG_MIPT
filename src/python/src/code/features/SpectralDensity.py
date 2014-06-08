import numpy as np
import logging
import time

from src.code.features import BasicFeature
import src.code.calculator as clc


class SpectralDensity(BasicFeature):
    type = 'NUMERIC'

    def __init__(self):
        pass

    def run(self, ecg, begin=200.0, end=500.0, normalized=True):
        y = ecg.getHighFreq()
        fq = ecg.getDataFrequency()
        assert end <= fq / 2.
        if normalized:
            y = clc.normalize(y, type='energy=1', sampling_fq=fq) # normed to RMS -> energy_speed = 1 / sec => energy = T
        if ecg._f is None or ecg._fft is None:
            tt = time.time()
            ecg._f = fq / 2. * np.linspace(0.0, 1.0, len(y)/2 + 1)
            ecg._fft = np.abs(np.fft.rfft(y)) / (fq / 2.) # energy(fft) == 2*energy(y)
            assert len(ecg._f) == len(ecg._fft), 'len f = %d, len fft = %d' % (len(ecg._f), len(ecg._fft))
            logging.info('FFT calculated in %.3f sec on data length %d', (time.time() - tt), len(y))
        croppedFft = np.asarray([y for x,y in zip(ecg._f, ecg._fft) if begin < x < end])
        res = np.sum(croppedFft ** 2) * (float(end-begin) / len(croppedFft))
        logging.info('Spectral density [%.2f, %.2f] calculated on %s data: %f' % (begin, end, 'normalized' if normalized else 'not normalized', res))
        return res