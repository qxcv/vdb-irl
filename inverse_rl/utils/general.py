import argparse
import time
import tensorflow as tf

import numpy as np


def flatten_list(lol):
    return [a for b in lol for a in b]


def argparse_layer_spec(spec_str):
    split_strip = [l for l in spec_str.split(',') if l.strip()]
    try:
        layer_counts = [int(c) for c in split_strip]
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            'layer counts must be comma-separated list of integers (like '
            '"64,32,64"), not "%s" (%s)' % (spec_str, ex))
    return layer_counts


class TrainingIterator(object):
    def __init__(self, itrs, heartbeat=float('inf')):
        self.itrs = itrs
        self.heartbeat_time = heartbeat
        self.__vals = {}

    def random_idx(self, N, size):
        return np.random.randint(0, N, size=size)

    @property
    def itr(self):
        return self.__itr

    @property
    def heartbeat(self):
        return self.__heartbeat

    @property
    def elapsed(self):
        assert self.heartbeat, 'elapsed is only valid when heartbeat=True'
        return self.__elapsed

    def itr_message(self):
        return '==> Itr %d/%d (elapsed:%.2f)' % (self.itr + 1, self.itrs,
                                                 self.elapsed)

    def record(self, key, value):
        if key in self.__vals:
            self.__vals[key].append(value)
        else:
            self.__vals[key] = [value]

    def pop(self, key):
        vals = self.__vals.get(key, [])
        del self.__vals[key]
        return vals

    def pop_mean(self, key):
        return np.mean(self.pop(key))

    def __iter__(self):
        prev_time = time.time()
        self.__heartbeat = False
        for i in range(self.itrs):
            self.__itr = i
            cur_time = time.time()
            if (cur_time - prev_time) > self.heartbeat_time or i == (
                    self.itrs - 1):
                self.__heartbeat = True
                self.__elapsed = cur_time - prev_time
                prev_time = cur_time
            yield self
            self.__heartbeat = False


def kl_loss(mean, logstd):
    # ugh having loss here is ugly, but not sure where else to put it.
    # formula here is (10) from AEVB paper.
    std = tf.exp(logstd)
    loss_parts = 0.5 * tf.reduce_sum(
        -1 - 2 * logstd + std**2 + mean**2, axis=-1)
    return tf.reduce_mean(loss_parts)
