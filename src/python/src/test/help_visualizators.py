import numpy as np

import matplotlib.pyplot as plt

from src.test import ECGDependentTest

class ECGVisualizator(ECGDependentTest):

    def test_plot_ecg(self):
        y_low = self.ecg().getLowFreq()
        y_high = self.ecg().getHighFreq()
        x = self.ecg().getTiming()
        __, p = plt.subplots(2)
        plt.plot(x,y_low,'r-')
        plt.plot(x,y_high,'g-')
        plt.show()