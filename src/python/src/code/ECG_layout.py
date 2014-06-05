# input: vector y - low frequency ECG
# output: list:
# [ #sorted by t
#     (<t_i - int>, <type_i - 'P'|'R'|'S'|'Q'...>) # y[t] - point of specified type
#     (<t_2>,      <type_2>)
#     (<t_3>,      <type_3>)
#     ...
# ]

import numpy as np
import matplotlib.pyplot as plt

import src.code.calculator as clc
import src.code.WaveletProcessor as WP

KNOWN_TYPES = {'P': {'plot_style': 'g^'},
               'R': {'plot_style': 'r^'}}


def dummyLayout(y):
    return [(1650, 'R')]


def generateSimpleLayout(y, sampling_frequency):
    y = clc.aline(y, sampling_frequency)
    main_fq = clc.getMainFrequency(y, sampling_frequency)
    beat_length = int(sampling_frequency / main_fq)
    layout = []
    R0 = np.argmax(y)
    R = R0
    layout.append((R, 'R'))
    # right
    while R + int(1.5 * beat_length) < len(y):
        interval = y[int(R + 0.5 * beat_length):int(R + 1.5 * beat_length)]
        R = int(R + 0.5 * beat_length) + np.argmax(interval)
        layout.append((R, 'R'))
    # left
    R= R0
    while R - int(1.5 * beat_length) >= 0:
        interval = y[int(R - 1.5 * beat_length):int(R - 0.5 * beat_length)]
        R = int(R - 1.5 * beat_length) + np.argmax(interval)
        layout.append((R, 'R'))
    return layout


def generate_modulus_maximum_layout(y, sampling_frequency):
    scale = int(0.008 * sampling_frequency)
    wt = WP.get_WT(y, scale)
    wt = clc.normalize(wt)
    mm = WP.get_mm_array(wt)
    mm = WP.filter_mm_array(mm, wt)
    THRESHOLD = 0.5



def drawLayout(x,y,layout):
    types = set([t for __, t in layout])
    layout_points = [(x[l[0]], y[l[0]], l[1]) for l in layout]
    plots = [x,y,'b-']
    for point_type in types:
        new_plot = [(x, y) for x, y, pType in layout_points if pType == point_type]
        x_l, y_l = map(list, zip(*new_plot))
        plots.extend([x_l, y_l, KNOWN_TYPES[point_type]['plot_style']])
    plt.plot(*plots)
    plt.show()