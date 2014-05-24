import logging
import numpy as np
import matplotlib.pyplot as plt
from src.code.calculator import aline

logging.basicConfig(level=logging.DEBUG)

def test_aline():
    POINTS = 8000
    SCALE = 200.
    sampling_fq = POINTS / SCALE
    x = np.array(np.linspace(0., 1., POINTS) * SCALE)
    y = np.sin(x) + np.sin(x / 20.) + 1.
    new_y = aline(y, sampling_fq)
    plt.plot(x, y, 'g-', x, new_y, 'r-')
    plt.show()

test_aline()