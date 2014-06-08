import numpy as np
import logging
import time

import src.code.calculator as clc
import src.code.WaveletProcessor as WP
from src.code.features import BasicFeature
from src.code.ECG_layout import generate_modulus_maximum_layout
import src.test.various_tests as vt

class LocalizedSpectralDensity(BasicFeature):
    type = 'NUMERIC'

    SCALE_POINTS = 10

    def __init__(self):
        pass

    def _integrate_wt(self, R_peaks, wt, sampling_fq, scale_range_size, slice_begin=0., slice_end=1.):
        energy = 0.
        for i in range(len(R_peaks)-1):
            beat_size = R_peaks[i+1] - R_peaks[i]
            begin = int(R_peaks[i] + slice_begin * beat_size)
            end = int(R_peaks[i] + slice_end * beat_size)
            wt_clipped = wt[:, begin:(end+1)]
            clip_time = float(end - begin) / sampling_fq
            if (float(beat_size) / sampling_fq) > (60. / 250.): # pulse > 250 bps
                energy += np.sum(wt_clipped ** 2) * (float(clip_time) / wt_clipped.shape[1])
            else:
                logging.error('Too small beat encountered: %f seconds', (float(beat_size) / sampling_fq))
        return energy * (float(scale_range_size) / wt.shape[0])

    def _calc_energy(self, y, R_peaks, sampling_fq, fq_begin, fq_end, slice_begin=0., slice_end=1., detalization=0.1):
        # as WT calc time ~scale^2 => adaptive scales should be applied to speed up calculations

        assert fq_end > fq_begin > 0
        assert 1. >= slice_end > slice_begin >= 0.

        scale_begin = np.round(WP.fq_to_ricker_scale(fq_end, sampling_fq), decimals=1)
        if scale_begin < 0.5:
            logging.error('Scale begin out of range: %f', scale_begin)
            scale_begin = 0.5
        scale_end = max(np.round(WP.fq_to_ricker_scale(fq_begin, sampling_fq), decimals=1), 0.5)
        if scale_end < 0.5:
            logging.error('Scale end out of range: %f', scale_end)
            raise ValueError('Frequency: [%f, %f], sampling_fq: %f', fq_begin, fq_end, sampling_fq)
        logging.debug('fq range [%f, %f] -> scale range [%f, %f]', fq_begin, fq_end, scale_begin, scale_end)

        TOLERANCE = 3 # OK for synthetic case
        scales = []
        finished = False
        interval_begin = scale_begin
        while not finished:
            step = max(0.1, np.round(detalization * interval_begin, decimals=1))
            # step = max(0.1, np.round(detalization * 2, decimals=1))
            interval_end = interval_begin * TOLERANCE
            if interval_end >= scale_end:
                finished = True
                interval_end = scale_end
            scale = np.linspace(interval_begin, interval_end, max(np.round((interval_end - interval_begin) / step) + 1, 2))
            scale = np.unique(np.round(scale, decimals=1))
            scales.append(scale)
            interval_begin = interval_end

        energy = 0.
        for scale in scales:
            wt = WP.get_WT(y, scale)
            # vt.draw_cwt_as_img(y, wt, scale)
            energy_on_scale = self._integrate_wt(R_peaks, wt, sampling_fq, scale[-1] - scale[0], slice_begin=slice_begin, slice_end=slice_end)
            logging.debug('Energy on scale [%f, %f]: %f', scale[0], scale[-1], energy_on_scale)
            energy += energy_on_scale
        return energy

    def run(self, ecg, beat_begin=0., beat_end=1., fq_begin=200, fq_end=400, normalized=True):
        assert 1. >= beat_end > beat_begin >= 0.
        assert fq_end > fq_begin > 0

        y_low = ecg.getLowFreq()
        y_high = ecg.getHighFreq()
        if normalized:
            y_high = clc.normalize(y_high)
        sampling_fq = ecg.getDataFrequency()
        assert fq_end < (sampling_fq / 2.) # cannot get higher resolution
        if ecg.layout is None:
            layout = generate_modulus_maximum_layout(y_low, sampling_fq)
            ecg.layout = layout
        R_peaks = [point[0] for point in ecg.layout if point[1] == 'R']
        if len(R_peaks) < 2:
            raise ValueError('Not enough beats detected: %d', len(R_peaks))
        energy = self._calc_energy(y_high, R_peaks, sampling_fq, fq_begin, fq_end, beat_begin, beat_end)
        return energy