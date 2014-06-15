import copy
import numpy as np

FQ_INTERVALS = [25., 50., 100., 150., 200., 250., 300., 400., 500., 750., 1000.]
FQ_RANGE = [0., 2000.]

VOID_DESCRIPTION = {'feature': None, 'name': None, 'options': None}
LSD_OPTIONS = {'normalized': True, 'beat_begin': 0.0, 'beat_end': 1.0,'fq_begin': 25., 'fq_end': 1000., 'calc_type': 'fft'}

def run_LSD_default_fq_given_interval(beat_begin, beat_end):
    run = []

    description_sample = copy.copy(VOID_DESCRIPTION)
    description_sample['feature'] = 'LocalizedSpectralDensity'
    description_sample['name'] = 'LocalizedSpectralDensity'

    options_sample = copy.copy(LSD_OPTIONS)
    options_sample['beat_begin'] = beat_begin
    options_sample['beat_end'] = beat_end

    for i in range(1, len(FQ_INTERVALS)):
        description = copy.copy(description_sample)
        description['name'] += str(i)
        options = copy.copy(options_sample)
        options['fq_begin'] = FQ_INTERVALS[i-1]
        options['fq_end'] = FQ_INTERVALS[i]
        description['options'] = options
        run.append(description)

    return run, 'py_out_LSD_default_fq_interval-[%.2f,%.2f].arff' % (beat_begin, beat_end)

def run_LSD_given_fq_ranges_count_and_given_interval(beat_begin, beat_end, ranges_count):
    run = []

    description_sample = copy.copy(VOID_DESCRIPTION)
    description_sample['feature'] = 'LocalizedSpectralDensity'
    description_sample['name'] = 'LocalizedSpectralDensity'

    options_sample = copy.copy(LSD_OPTIONS)
    options_sample['beat_begin'] = beat_begin
    options_sample['beat_end'] = beat_end

    fq_intervals = np.linspace(FQ_RANGE[0], FQ_RANGE[1], ranges_count+1)

    for i in range(1, len(fq_intervals)):
        description = copy.copy(description_sample)
        description['name'] += str(i)
        options = copy.copy(options_sample)
        options['fq_begin'] = fq_intervals[i-1]
        options['fq_end'] = fq_intervals[i]
        description['options'] = options
        run.append(description)

    return run, 'py_out_LSD_%d_fq_ranges__beat_interval-[%.2f,%.2f].arff' % (ranges_count, beat_begin, beat_end)