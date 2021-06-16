# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import math
import random

inf = float('inf')

def linspace(start, stop, num=50, endpoint=True):
    """linspace from a to b with n entries."""
    step = (stop - start) / (num - endpoint)
    return [start + step*i for i in range(num)]

def log_linspace(a, b, n):
    """linspace from a to b with n entries over log scale."""
    points = linspace(math.log(a), math.log(b), num=n)
    return [math.exp(x) for x in points]

def log_normalize(log_weights):
    """Return log of the sum of exponentials of input list divided by sum."""
    Z = logsumexp(log_weights)
    return [x - Z for x in log_weights]

def log_choices(population, log_weights, k=1, prng=None):
    """Draw from a population given a list of log probabilities."""
    log_weights_normalized = log_normalize(log_weights)
    weights = [math.exp(w) for w in log_weights_normalized]
    return (prng or random).choices(population, weights, k=k)

def logsumexp(array):
    """Return log of the sum of exponentials of input elements."""
    if len(array) == 0:
        return float('-inf')

    # m = +inf means addends are all +inf, hence so are sum and log.
    # m = -inf means addends are all zero, hence so is sum, and log is
    # -inf.  But if +inf and -inf are among the inputs, or if input is
    # NaN, let the usual computation yield a NaN.
    m = max(array)
    if math.isinf(m) \
            and min(array) != -m \
            and all(not math.isnan(a) for a in array):
        return m

    # Since m = max{a_0, a_1, ...}, it follows that a <= m for all a,
    # so a - m <= 0; hence exp(a - m) is guaranteed not to overflow.
    return m + math.log(sum(math.exp(a - m) for a in array))

def logmeanexp(array):
    """Return log of the mean of exponentials of input elements."""
    if len(array) == 0:
        return -inf

    # Treat -inf values as log 0 -- they contribute zero to the sum in
    # logsumexp, but one to the count.
    #
    # If we pass -inf values through to logsumexp, and there are also
    # +inf values, then we get NaN -- but if we had averaged exp(-inf)
    # = 0 and exp(+inf) = +inf, we would sensibly get +inf, whose log
    # is still +inf, not NaN.  So strip -inf values first.
    #
    # Can't say `a > -inf' because that excludes NaNs, but we want to
    # include them so they propagate.
    noninfs = [a for a in array if not a == -inf]

    # probs = map(exp, array)
    # log(mean(probs))
    #   = log(sum(probs) / len(probs))
    #   = log(sum(probs)) - log(len(probs))
    #   = log(sum(map(exp, array))) - log(len(array))
    #   = logsumexp(array) - log(len(array))
    return logsumexp(noninfs) - math.log(len(array))
