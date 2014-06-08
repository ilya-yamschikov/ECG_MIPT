import wave
import struct
import logging
import os

import numpy as np

from src.code.calculator import filterSignal


def get_filename_without_extension(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class MouseECG:
    NORMAL_FILES = []

    def __init__(self, fileName):
        waveObj = wave.open(fileName, 'r')
        framesCount = waveObj.getnframes()
        self.frequency = waveObj.getframerate()
        channelsCount = waveObj.getnchannels()

        assert channelsCount == 2, 'File with %d channels found. Only 2 channel files are supported now.' % channelsCount

        logging.info('File "%s" loaded' % fileName)
        logging.info('Parameters: [channels: %s; frequency: %s; frames: %s; compression: %s; sample width: %s]' % (channelsCount, self.frequency, framesCount, waveObj.getcompname(), waveObj.getsampwidth()))
        raw_data = waveObj.readframes(framesCount)
        mixed_data = list(struct.unpack('=%dh' % (framesCount * channelsCount), raw_data))

        self._inverted = get_filename_without_extension(fileName) not in self.NORMAL_FILES
        self.layout = None

        self.data = []
        for i in xrange(channelsCount):
            self.data.append(mixed_data[i::channelsCount])
        self.timing = np.arange(framesCount) / np.float32(self.frequency)

    def getClass(self):
        return 'HEALTHY'

    def getDataFrequency(self):
        return self.frequency

    def getTiming(self):
        return self.timing

    def getLowFreq(self):
        res = np.array(self.data[1])
        return res if not self._inverted else -res

    def getHighFreq(self):
        res = np.array(self.data[0])
        return res if not self._inverted else -res

class PTB_ECG:
    INVERTED_FILES = ['s0010_re', 's0014lre', 's0016lre', 's0029lre', 's0043lre', 's0050lre', 's0054lre', 's0059lre', 's0082lre', 's0062lre']
    SPLIT_FREQUENCY = 200.

    def __init__(self, fileName):
        descrFile = open(fileName + '.descr', 'r')
        self.Class = descrFile.readline()
        descrFile.close()
        dataFile = open(fileName + '.csv', 'r')
        self._inverted = get_filename_without_extension(fileName) in self.INVERTED_FILES
        self.layout = None
        x = []
        y = []
        for line in dataFile:
            num = line.split(',')
            assert len(num) == 2
            x.append(float(num[0]))
            y.append(float(num[1]))
        self.timing = np.asarray(x)
        self.y = np.asarray(y)
        self.frequency = 1 / (self.timing[1] - self.timing[0])
        self.lowFreq = filterSignal(self.y, self.SPLIT_FREQUENCY, self.frequency, filterType='lowpass')
        self.highFreq = filterSignal(self.y, self.SPLIT_FREQUENCY, self.frequency, filterType='highpass')

    def getClass(self):
        return self.Class

    def getTiming(self):
        return self.timing

    def getDataFrequency(self):
        return self.frequency

    def getLowFreq(self):
        res = self.lowFreq
        return res if not self._inverted else -res

    def getHighFreq(self):
        res = self.highFreq
        return res if not self._inverted else -res
