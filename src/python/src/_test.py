import matplotlib.pyplot as plt
import numpy as np
import calculator as clc

from ECG_loader import MouseECG
ecg = MouseECG(r'..\data\new_data\1_1.wav')
# x = ecg.getTiming()
# y = ecg.getHighFreq()

x = np.arange(5000) / 25.0
y = np.sin(x) * 3.3


# y = clc.normalize(y)
fft = np.fft.rfft(y) / (0.5*len(y))
fft = np.absolute(fft)
f = ecg.getDataFrequency()/2 * np.linspace(0.0, 1.0, len(x)/2 + 1)

plt.plot(x,y)
plt.show()
plt.plot(f, fft)
plt.show()
print str(fft)
print 'that\'s all'