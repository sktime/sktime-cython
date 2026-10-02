# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython MultiRocketMultivariate kernels.

Ahead-of-time compiled ports of sktime's ``_multirocket_multi_numba`` kernels
(``_fit_biases`` and ``_transform``). Same math, no numba JIT
warmup. TODO.
"""

import numpy as np

cimport numpy as cnp
cimport cython
from libc.stdlib cimport free, malloc
from libc.stdint cimport int64_t
from cython cimport floating

cnp.import_array()

# combinations(range(9), 3) -> 84 kernels, flattened to 252 int32.
cdef int[252] _IDX
cdef int _fill_idx():
    cdef int n = 0, i, j, k
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                _IDX[3 * n] = i
                _IDX[3 * n + 1] = j
                _IDX[3 * n + 2] = k
                n += 1
    return n
cdef int _NUM_KERNELS = _fill_idx()


cdef void _one(
    floating[:, :, ::1] X,
    int ex,
    int n_columns,
    int n_timepoints,
    int raw_n_timepoints,
    int[::1] num_channels_per_combination,
    int[::1] channel_indices,
    int[::1] dilations,
    int[::1] num_features_per_dilation,
    float[::1] biases,
    int num_features,
    int64_t feature_offset,
    bint advance_combination,
    float* C_alpha,
    float* C_gamma,
    float* C,
    float[:, ::1] feat,
) noexcept nogil:
    """One representation (raw or differenced) for one instance.

    Writes 4 features per kernel at ``feature_offset + fis + fc`` and every
    ``num_features`` thereafter. Buffers are caller-allocated and reused.
    """
    cdef int num_kernels = _NUM_KERNELS
    cdef int num_dilations = dilations.shape[0]
    cdef int csize = n_columns * n_timepoints

    cdef int di, ki, g, c, t, ch
    cdef int padding, padding0, padding1, nfd, dilation
    cdef int start, end, base, combination_index
    cdef int fis, fie, ncc, ncs, nce
    cdef int i0, i1, i2, fc, n_valid, j
    cdef float* _c
    cdef int ppv, last_val, stretch, max_stretch
    cdef int64_t mean_index, feature_index
    cdef float bias
    cdef floating x
    cdef double mean

    fis = 0
    ncs = 0
    combination_index = 0
    for di in range(num_dilations):
        padding0 = di % 2
        dilation = dilations[di]
        padding = ((9 - 1) * dilation) // 2
        nfd = num_features_per_dilation[di]

        # C_alpha = A = -X ; C_gamma = 0 ; C_gamma[4] = G = 3X
        for c in range(n_columns):
            base = c * n_timepoints
            for t in range(n_timepoints):
                x = X[ex, c, t]
                C_alpha[base + t] = -x
                C_gamma[4 * csize + base + t] = x + x + x
        for g in range(9):
            if g == 4:
                continue
            for c in range(n_columns):
                base = g * csize + c * n_timepoints
                for t in range(n_timepoints):
                    C_gamma[base + t] = 0.0

        # gamma_index 0..3. The caller passes raw_n_timepoints == n_timepoints
        # for the corrected transform; in the original implementation the
        # differenced pass receives the undifferenced length here, which sits
        # its windows one sample further left (sktime#11291).
        end = raw_n_timepoints - padding
        # `end` stays within the row (and `n_valid` below stays positive) because
        # dilation >= 1 and 8 * dilation < n_timepoints; boundscheck is off here,
        # so the Python layer enforces both in ``_common._check_dilations``.
        for g in range(4):
            if end > 0:
                for c in range(n_columns):
                    base = c * n_timepoints
                    for t in range(end):
                        C_alpha[base + n_timepoints - end + t] += -X[ex, c, t]
                        C_gamma[g * csize + base + n_timepoints - end + t] = (
                            3.0 * X[ex, c, t]
                        )
            end += dilation

        # gamma_index 5..8
        start = dilation
        for g in range(5, 9):
            if start < n_timepoints:
                for c in range(n_columns):
                    base = c * n_timepoints
                    for t in range(n_timepoints - start):
                        C_alpha[base + t] += -X[ex, c, start + t]
                        C_gamma[g * csize + base + t] = (
                            3.0 * X[ex, c, start + t]
                        )
            start += dilation

        for ki in range(num_kernels):
            fie = fis + nfd
            ncc = num_channels_per_combination[combination_index]
            nce = ncs + ncc
            padding1 = (padding0 + ki) % 2
            i0 = _IDX[3 * ki]
            i1 = _IDX[3 * ki + 1]
            i2 = _IDX[3 * ki + 2]

            # C[t] = sum over the combination's channels
            for t in range(n_timepoints):
                C[t] = 0.0
            for ch in range(ncs, nce):
                c = channel_indices[ch]
                base = c * n_timepoints
                for t in range(n_timepoints):
                    C[t] += (
                        C_alpha[base + t]
                        + C_gamma[i0 * csize + base + t]
                        + C_gamma[i1 * csize + base + t]
                        + C_gamma[i2 * csize + base + t]
                    )

            if padding1 == 0:
                for fc in range(nfd):
                    bias = biases[fis + fc]
                    ppv = 0
                    last_val = 0
                    max_stretch = 0
                    mean_index = 0
                    mean = 0.0

                    for j in range(n_timepoints):
                        if C[j] > bias:
                            ppv += 1
                            mean_index += j
                            mean += C[j] + bias
                        elif C[j] < bias:
                            stretch = j - last_val
                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = n_timepoints - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    feature_index = feature_offset + fis + fc
                    feat[ex, feature_index] = <float>ppv / n_timepoints
                    feature_index += num_features
                    feat[ex, feature_index] = <float>max_stretch
                    feature_index += num_features
                    feat[ex, feature_index] = mean / ppv if ppv > 0 else 0.0
                    feature_index += num_features
                    feat[ex, feature_index] = <float>mean_index / ppv if ppv > 0 else -1.0
            else:
                _c = C + padding
                n_valid = n_timepoints - 2 * padding

                for fc in range(nfd):
                    bias = biases[fis + fc]
                    ppv = 0
                    last_val = 0
                    max_stretch = 0
                    mean_index = 0
                    mean = 0

                    for j in range(n_valid):
                        if _c[j] > bias:
                            ppv += 1
                            mean_index += j
                            mean += _c[j] + bias
                        elif _c[j] < bias:
                            stretch = j - last_val
                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = n_valid - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    feature_index = feature_offset + fis + fc
                    feat[ex, feature_index] = <float>ppv / n_valid
                    feature_index += num_features
                    feat[ex, feature_index] = <float>max_stretch
                    feature_index += num_features
                    feat[ex, feature_index] = mean / ppv if ppv > 0 else 0.0
                    feature_index += num_features
                    feat[ex, feature_index] = (
                        <float>mean_index / ppv if ppv > 0 else -1.0
                    )

            fis = fie
            # the original implementation does not advance these in the
            # differenced pass, so every kernel there reuses the first
            # combination's channel selection (sktime#11291).
            if advance_combination:
                combination_index += 1
                ncs = nce


def transform(
    floating[:,:,::1] X,
    floating[:,:,::1] X1,
    int[::1] num_channels_per_combination,
    int[::1] channel_indices,
    int[::1] dilations,
    int[::1] num_features_per_dilation,
    float[::1] biases,
    int[::1] num_channels_per_combination1,
    int[::1] channel_indices1,
    int[::1] dilations1,
    int[::1] num_features_per_dilation1,
    float[::1] biases1,
    bint original_implementation,
):
    """Port of ``_transform``. X is (n_instances, n_columns, n_timepoints);
        X1 is (n_instances, n_columns, n_timepoints-1).

    ``original_implementation`` reproduces sktime's reference behaviour for the
    differenced pass: windows sized from the undifferenced length, cursors not
    advanced, and the base pass's channel selection reused.
    """
    cdef int n_instances = X.shape[0]
    cdef int n_columns = X.shape[1]
    cdef int n_timepoints = X.shape[2]

    cdef int num_kernels = _NUM_KERNELS
    cdef int num_dilations = dilations.shape[0]
    cdef int num_dilations1 = dilations1.shape[0]

    cdef int total_fpd = 0
    cdef int d
    for d in range(num_dilations):
        total_fpd += num_features_per_dilation[d]
    cdef int total_fpd1 = 0
    cdef int d1
    for d1 in range(num_dilations1):
        total_fpd1 += num_features_per_dilation1[d1]
    cdef int num_features = num_kernels * total_fpd
    cdef int num_features1 = num_kernels * total_fpd1

    cdef int64_t num_total_features = <int64_t>(num_features + num_features1) * 4
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] features = np.zeros(
        (n_instances, num_total_features), dtype=np.float32
    )
    cdef int64_t num_features_per_transform = num_total_features >> 1
    cdef float[:, ::1] feat = features

    cdef int csize = n_columns * n_timepoints
    # Per-instance work buffers (single-threaded; reused across dilations).
    cdef float* C_alpha = <float*>malloc(csize * sizeof(float))
    cdef float* C_gamma = <float*>malloc(9 * csize * sizeof(float))
    cdef float* C = <float*>malloc(n_timepoints * sizeof(float))
    if C_alpha == NULL or C_gamma == NULL or C == NULL:
        free(C_alpha); free(C_gamma); free(C)
        raise MemoryError()

    cdef int ex

    try:
      with nogil:
        for ex in range(n_instances):
            _one(
                X, ex, n_columns, n_timepoints, n_timepoints,
                num_channels_per_combination, channel_indices,
                dilations, num_features_per_dilation, biases,
                num_features, 0, True,
                C_alpha, C_gamma, C, feat,
            )
            if original_implementation:
                _one(
                    X1, ex, n_columns, n_timepoints - 1, n_timepoints,
                    num_channels_per_combination, channel_indices,
                    dilations1, num_features_per_dilation1, biases1,
                    num_features, num_features_per_transform, False,
                    C_alpha, C_gamma, C, feat,
                )
            else:
                _one(
                    X1, ex, n_columns, n_timepoints - 1, n_timepoints - 1,
                    num_channels_per_combination1, channel_indices1,
                    dilations1, num_features_per_dilation1, biases1,
                    num_features, num_features_per_transform, True,
                    C_alpha, C_gamma, C, feat,
                )

    finally:
        free(C_alpha)
        free(C_gamma)
        free(C)

    return features


def fit_biases(
    floating[:,:,::1] X,
    int[::1] num_channels_per_combination,
    int[::1] channel_indices,
    int[::1] dilations,
    int[::1] num_features_per_dilation,
    int[::1] instance_indices,
):
    """Build per-combination convolution output C, summed over channels.

    Port of the convolution-building half of ``_fit_biases``. Returns a
    ``(num_combinations, n_timepoints)`` float32 array; the caller applies
    ``np.quantile`` per combination to obtain biases. ``instance_indices`` holds
    the ``np.random.randint(n_instances)`` draw for each combination, computed by
    the caller to reproduce the numba random sequence exactly.
    """
    cdef int n_timepoints = X.shape[2]

    cdef int num_kernels = _NUM_KERNELS
    cdef int num_dilations = dilations.shape[0]
    cdef int num_combinations = num_kernels * num_dilations

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out = np.zeros(
        (num_combinations, n_timepoints), dtype=np.float32
    )
    cdef float[:, ::1] outv = out

    # 9 gamma planes x n_timepoints for a single (instance, channel) row.
    cdef float* C_alpha = <float*>malloc(n_timepoints * sizeof(float))
    cdef float* C_gamma = <float*>malloc(9 * n_timepoints * sizeof(float))
    if C_alpha == NULL or C_gamma == NULL:
        free(C_alpha); free(C_gamma)
        raise MemoryError()

    cdef int di, ki, g, t, ch, ex, c
    cdef int padding, dilation, start, end
    cdef int comb, ncs, nce, ncc
    cdef int i0, i1, i2
    cdef floating x

    try:
        comb = 0
        ncs = 0
        for di in range(num_dilations):
            dilation = dilations[di]
            padding = ((9 - 1) * dilation) // 2

            for ki in range(num_kernels):
                ncc = num_channels_per_combination[comb]
                nce = ncs + ncc
                ex = instance_indices[comb]
                i0 = _IDX[3 * ki]
                i1 = _IDX[3 * ki + 1]
                i2 = _IDX[3 * ki + 2]

                for t in range(n_timepoints):
                    outv[comb, t] = 0.0

                # Accumulate C = C_alpha + C_gamma[i0,i1,i2], summed over channels.
                for ch in range(ncs, nce):
                    c = channel_indices[ch]

                    # C_alpha = A = -X ; C_gamma = 0 ; C_gamma[4] = G = 3X
                    for t in range(n_timepoints):
                        x = X[ex, c, t]
                        C_alpha[t] = -x
                        C_gamma[4 * n_timepoints + t] = x + x + x
                    for g in range(9):
                        if g == 4:
                            continue
                        for t in range(n_timepoints):
                            C_gamma[g * n_timepoints + t] = 0.0

                    end = n_timepoints - padding
                    for g in range(4):
                        if end > 0:
                            for t in range(end):
                                C_alpha[n_timepoints - end + t] += -X[ex, c, t]
                                C_gamma[g * n_timepoints + n_timepoints - end + t] = (
                                    3.0 * X[ex, c, t]
                                )
                        end += dilation

                    start = dilation
                    for g in range(5, 9):
                        if start < n_timepoints:
                            for t in range(n_timepoints - start):
                                C_alpha[t] += -X[ex, c, start + t]
                                C_gamma[g * n_timepoints + t] = (
                                    3.0 * X[ex, c, start + t]
                                )
                        start += dilation

                    for t in range(n_timepoints):
                        outv[comb, t] += (
                            C_alpha[t]
                            + C_gamma[i0 * n_timepoints + t]
                            + C_gamma[i1 * n_timepoints + t]
                            + C_gamma[i2 * n_timepoints + t]
                        )

                comb += 1
                ncs = nce
    finally:
        free(C_alpha)
        free(C_gamma)

    return out
