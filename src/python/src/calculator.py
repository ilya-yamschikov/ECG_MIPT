import math
import numpy as np

def normalize(x):
    return x / np.mean(np.absolute(x))

def RMS(x):
    return math.sqrt(x.dot(x) / len(x))