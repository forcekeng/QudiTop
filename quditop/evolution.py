"""Time evolution of quantum state."""

from typing import List, Tuple

import torch
from torch import Tensor


def evolution(op_mat: Tensor, qs: Tensor, target_indices: List[int]) -> Tensor:
    """Get the new quantum state after applying specific operation(gate or matrix).
    Refer: `https://pyquil-docs.rigetti.com/en/stable/_modules/pyquil/simulation/_numpy.html`

    Args:
        op_mat: The operation matrix that change the quantum state.
        qs: Current quantum state.
        target_indices: The qudits that `op_mat` acts on.

    Returns:
        The new quantum state.
    """
    k = len(target_indices)
    d = len(qs.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple(data_indices[q] for q in target_indices)
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, target_indices):
        output_indices[t] = w
    return torch.einsum(op_mat, input_indices, qs, data_indices, output_indices)


def evolution_complex(op_mat: Tuple, qs: Tuple, target_indices: List[int]) -> Tensor:
    """Get the new quantum state after applying specific operation(gate or matrix).
    Since the auto-difference of complex number is not supported in PyTorch, Here just decompose the complex
    matrix as a tuple (real, imag) which represents the real part and imaginary part respectively.

    Args:
        op_mat: The operation matrix that change the quantum state.
        qs: Current quantum state.
        target_indices: The qudits that `op_mat` acts on.

    Returns:
        The new quantum state.
    """
    op_mat_real, op_mat_imag = op_mat
    qs_real, qs_imag = qs
    qs2_real = evolution(op_mat_real, qs_real, target_indices) \
        - evolution(op_mat_imag, qs_imag, target_indices)
    qs2_imag = evolution(op_mat_real, qs_imag, target_indices) \
        + evolution(op_mat_imag, qs_real, target_indices)
    return qs2_real, qs2_imag
