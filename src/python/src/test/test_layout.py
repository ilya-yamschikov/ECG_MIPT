import logging
from src.code.ECG_layout import *
from src.code.ECG_loader import PTB_ECG

logging.basicConfig(level=logging.DEBUG)

def test_simple_layout(ecg):
    x = ecg.getTiming()
    y = ecg.getLowFreq()
    sampling_frequency = ecg.getDataFrequency()
    layout = generateSimpleLayout(y, sampling_frequency)
    drawLayout(x, y, layout)

ecg = PTB_ECG(r'..\..\..\..\data\ptb_database_csv\s0001_re')
test_simple_layout(ecg)