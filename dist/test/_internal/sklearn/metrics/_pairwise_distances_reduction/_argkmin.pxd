# WARNING: Do not edit this file directly.
# It is automatically generated from 'sklearn\\metrics\\_pairwise_distances_reduction\\_argkmin.pxd.tp'.
# Changes must be made there.

from ...utils._typedefs cimport intp_t, float64_t

from ._base cimport BaseDistancesReduction64
from ._middle_term_computer cimport MiddleTermComputer64

cdef class ArgKmin64(BaseDistancesReduction64):
    """float64 implementation of the ArgKmin."""

    cdef:
        intp_t k

        intp_t[:, ::1] argkmin_indices
        float64_t[:, ::1] argkmin_distances

        # Used as array of pointers to private datastructures used in threads.
        float64_t ** heaps_r_distances_chunks
        intp_t ** heaps_indices_chunks


cdef class EuclideanArgKmin64(ArgKmin64):
    """EuclideanDistance-specialisation of ArgKmin64."""
    cdef:
        MiddleTermComputer64 middle_term_computer
        const float64_t[::1] X_norm_squared
        const float64_t[::1] Y_norm_squared

        bint use_squared_distances

from ._base cimport BaseDistancesReduction32
from ._middle_term_computer cimport MiddleTermComputer32

cdef class ArgKmin32(BaseDistancesReduction32):
    """float32 implementation of the ArgKmin."""

    cdef:
        intp_t k

        intp_t[:, ::1] argkmin_indices
        float64_t[:, ::1] argkmin_distances

        # Used as array of pointers to private datastructures used in threads.
        float64_t ** heaps_r_distances_chunks
        intp_t ** heaps_indices_chunks


cdef class EuclideanArgKmin32(ArgKmin32):
    """EuclideanDistance-specialisation of ArgKmin32."""
    cdef:
        MiddleTermComputer32 middle_term_computer
        const float64_t[::1] X_norm_squared
        const float64_t[::1] Y_norm_squared

        bint use_squared_distances
