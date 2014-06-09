import wave
import struct
import logging
import os

import numpy as np

from src.code.calculator import filterSignal


def get_filename_without_extension(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class BasicECG:
    def __init__(self):
        self.layout = None
        self._f = None
        self._fft = None

    def _invert(self, res):
        return res if not self._inverted else -res

class MouseECG(BasicECG):
    NORMAL_FILES = ['16_5', '18_1']
    CLASSES_MAPPING = {'DO1': ['1_1', '2_1', '3_1', '5_1', '6_1', '7_1', '8_1', '9_1', '10_1', '11_1', '12_1', '13_1', '16_1', '17_1', '18_1', '19_1', '20_1', '21_1', '22_1', '23_1'],
                   'T1': ['1_2', '2_2', '3_2', '5_2', '6_2', '7_2', '8_2', '9_2', '10_2', '11_2', '12_2', '13_2', '16_2', '17_2', '18_2', '19_2', '20_2', '21_2', '22_2', '23_2'],
                   'I1': ['1_3', '3_3', '5_3', '9_3', '10_3', '13_3', '16_3', '17_3', '18_3', '19_3', '20_3', '21_3', '22_3', '23_3'],
                   'I2': ['6_3'],
                   'I3': ['7_3', '12_3'],
                   'I5': ['2_3', '4_1', '5_4', '8_3', '9_4', '11_3', '16_4', '18_4', '19_4', '20_4', '21_4'],
                   'I7': ['7_4', '17_4'],
                   'I10': ['1_4', '2_4', '3_4', '5_5', '6_4', '8_4', '10_4', '11_4', '12_4', '13_4', '16_5', '17_5', '18_5', '19_5', '20_5', '21_5', '22_4', '23_4'],
                   'I20': ['5_6', '6_5', '9_5', '10_5', '16_6', '17_6', '18_6', '21_6', '22_5', '23_5'],
                   'I25': ['7_5'],
                   'I30': ['1_5', '3_5', '16_7', '18_7', '19_6', '21_7'],
                   'I40': ['1_6', '2_5', '3_6', '4_2', '5_7', '7_6', '9_6', '10_6', '13_5', '16_8', '17_7', '18_8', '19_7', '20_6', '21_8'],
                   'I42': ['8_5', '12_5'],
                   'I45': ['11_5'],
                   'RP': ['1_7', '4_3', '5_8', '7_7', '11_6', '12_6', '13_6', '17_8', '18_9', '19_8', '20_7', '21_9'],
                   'RP1': ['2_6', '8_6', '9_7', '10_7'],
                   'RP5': ['1_8', '3_7'],
                   'RP30': ['7_8']}
    Classes = CLASSES_MAPPING.keys()
    PULSE_NORM = {'interval': [300., 800], 'peak width': 0.004}

    def __init__(self, fileName):
        BasicECG.__init__(self)
        self.animal = 'mouse'
        waveObj = wave.open(fileName, 'r')
        framesCount = waveObj.getnframes()
        self.frequency = waveObj.getframerate()
        channelsCount = waveObj.getnchannels()
        self.Class = self.resolveClass(fileName)

        assert channelsCount == 2, 'File with %d channels found. Only 2 channel files are supported now.' % channelsCount

        logging.info('File "%s" loaded' % fileName)
        logging.info('Parameters: [channels: %s; frequency: %s; frames: %s; compression: %s; sample width: %s]' % (channelsCount, self.frequency, framesCount, waveObj.getcompname(), waveObj.getsampwidth()))
        raw_data = waveObj.readframes(framesCount)
        mixed_data = list(struct.unpack('=%dh' % (framesCount * channelsCount), raw_data))

        self._inverted = get_filename_without_extension(fileName) not in self.NORMAL_FILES

        self.data = []
        for i in xrange(channelsCount):
            self.data.append(mixed_data[i::channelsCount])
        self.lowFreq = np.array(self.data[1], dtype=np.float64)
        self.highFreq = np.array(self.data[0], dtype=np.float64)
        self.timing = np.arange(framesCount) / np.float32(self.frequency)
        if len(self.lowFreq) % 2 != 0:
            self.lowFreq = self.lowFreq[:-1]
            self.highFreq = self.highFreq[:-1]
            self.timing = self.timing[:-1]

    def resolveClass(self, fileName):
        filename_wo_extension = os.path.basename(os.path.splitext(fileName)[0])
        for _class, _files in self.CLASSES_MAPPING.iteritems():
            if filename_wo_extension in _files:
                return _class
        return 'UNKNOWN'

    def getClass(self):
        return self.Class

    def getDataFrequency(self):
        return self.frequency

    def getTiming(self):
        return self.timing

    def getLowFreq(self):
        return self._invert(self.lowFreq)

    def getHighFreq(self):
        return self._invert(self.highFreq)

class PTB_ECG(BasicECG):
    INVERTED_FILES = ['s0010_re', 's0014lre', 's0016lre', 's0029lre', 's0043lre', 's0050lre', 's0054lre', 's0059lre', 's0082lre', 's0062lre']
    SPLIT_FREQUENCY = 200.
    Classes = ['HEALTHY', 'MI']
    PULSE_NORM = {'interval': [40., 150], 'peak width': 0.008}

    def __init__(self, fileName):
        BasicECG.__init__(self)
        self.animal = 'human'
        descrFile = open(fileName + '.descr', 'r')
        self.Class = descrFile.readline()
        descrFile.close()
        dataFile = open(fileName + '.csv', 'r')
        self._inverted = get_filename_without_extension(fileName) in self.INVERTED_FILES
        x = []
        y = []
        for line in dataFile:
            num = line.split(',')
            assert len(num) == 2
            x.append(float(num[0]))
            y.append(float(num[1]))
        self.timing = np.asarray(x)
        self.y = np.asarray(y)
        # prime numbers problem - if len(y) is prime
        # fft is calculated very slow (may be ~100 times slower)
        if len(y) % 2 != 0:
            self.timing = self.timing[:-1]
            self.y = self.y[:-1]
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
        return self._invert(self.lowFreq)

    def getHighFreq(self):
        return self._invert(self.highFreq)

    def getSignal(self):
        return self._invert(self.y)
