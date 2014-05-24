import logging
import src.code.calculator as clc
from src.code.features import BasicFeature

class RMS(BasicFeature):
    type = 'NUMERIC'

    def __init__(self):
        pass

    def run(self, ecg, normalized=True):
        yHi = ecg.getHighFreq()
        y = clc.normalize(yHi) if normalized else yHi
        rms = clc.RMS(y)
        logging.info('Calculated %s RMS = %.4f' % ('normalized' if normalized else 'not normalized',rms))
        return rms