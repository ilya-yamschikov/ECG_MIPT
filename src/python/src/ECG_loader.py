import wave
import struct
import numpy as np
import logging

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