import matplotlib.pyplot as plt
import numpy as np
import calculator as clc

from ECG_loader import MouseECG, PTB_ECG
ecg = MouseECG(r'..\..\..\data\new_data\1_1.wav')
# x = ecg.getTiming()
# y = ecg.getHighFreq()

x = np.arange(8000) / 150.0
y = np.sin(2 * np.pi * x) + 0.1 * np.sin(2 * np.pi * 10 * x)

# y = clc.normalize(y)
fft = np.fft.rfft(y) / (0.5*len(y))
fft = np.absolute(fft)
f = ecg.getDataFrequency()/2 * np.linspace(0.0, 1.0, len(x)/2 + 1)

plt.plot(x,y)
plt.show()
plt.plot(f, fft)
plt.show()

ptb_ecg = PTB_ECG(r'..\..\..\data\ptb_database_csv\s0001_re')
y1 = ptb_ecg._filterSignal(y, 5, samplingFrequency=150, filterType='lowpass')
plt.plot(x, y1, 'r-', x, y, 'b-')
plt.show()

print 'that\'s all'