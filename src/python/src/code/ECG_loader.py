import wave
import struct
import logging

import numpy as np

from src.code.calculator import filterSignal


class MouseECG:
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
        return np.array(self.data[1])

    def getHighFreq(self):
        return np.array(self.data[0])

class PTB_ECG:
    SPLIT_FREQUENCY = 200.

    def __init__(self, fileName):
        descrFile = open(fileName + '.descr', 'r')
        self.Class = descrFile.readline()
        descrFile.close()
        dataFile = open(fileName + '.csv', 'r')
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
        return self.lowFreq

    def getHighFreq(self):
        return self.highFreq
