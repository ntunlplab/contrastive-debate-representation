import numpy
import torch
from functools import reduce

def get_params(model):
    excludes = get_excludes(model)
    parameters = { }
    for n, p in model.named_parameters():
        if n not in excludes:
            parameters[n] = p.detach().cpu().numpy().copy()
    return parameters

def get_excludes(model):
    names = [name for name, _ in model.named_parameters() \
             if ('lf.' in name)]# and ('lf.pooler.dense' not in name)]
    return names

def stats(param1, param2, zero_threshold=1e-8):
    keys = list(param1.keys())
    averages = { }
    stds = { }
    maxes = { }
    zero_counts = { }
    for key in keys:
        arr1 = param1[key]
        arr2 = param2[key]
        # element-wise diff
        diff = numpy.abs(numpy.subtract(arr1, arr2))
        # elment-wise diff is zero
        zero_count = (diff < zero_threshold).sum()

        averages[key] = numpy.mean(diff)
        stds[key] = numpy.std(diff)
        maxes[key] = numpy.max(diff)
        zero_counts[key] = zero_count
    return keys, averages, stds, maxes, zero_counts
    
def save_report(param1, param2, path):
    stats_keys, stats_averages, stats_stds, stats_maxes, stats_zero_counts = stats(param1, param2)
    with open(path, 'w') as f:
        for key in stats_keys:
            f.write('[%s]\n' % key)
            f.write('\tMean:\t{}\n'.format(stats_averages[key]))
            f.write('\tStd:\t{}\n'.format(stats_stds[key]))
            f.write('\tMax:\t{}\n'.format(stats_maxes[key]))
            f.write('\tZeros:\t{}\n'.format(stats_zero_counts[key]))
            f.write('-------------------\n\n')
