from src.code.ECG_layout import *

ecg = MouseECG(r'..\..\..\data\new_data\1_1.wav')
x = ecg.getTiming()
y_l = ecg.getLowFreq()
y = ecg.getHighFreq()

layout = generateSimpleLayout(y_l)
drawLayout(x,y,layout)

print 'that\'s all'