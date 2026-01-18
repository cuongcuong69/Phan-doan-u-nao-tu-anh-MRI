# -*- coding: utf-8 -*-
"""
Ramp-up/down cho hệ số consistency, theo UA-MT/MT.
"""
import numpy as np


def sigmoid_rampup(current, rampup_length):
    """
    current: tỉ lệ tiến trình (0..1) hoặc số step (0..rampup_length)
    rampup_length: nếu là tỉ lệ (0..1) → current nên cũng là tỉ lệ
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1.0))
