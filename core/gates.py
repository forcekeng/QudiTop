"""Qudits gates."""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from global_var import DTYPE, DEFAULT_VALUE, DEFAULT_PARAM_NAME
from utils import bprint, str_special, str_ket


def ket(i, dim):
    """Get the numerical column vector.

    Args:
        i: Value of ket.
        dim: Dimension of qudits.
    """
    vec = torch.zeros((dim, 1), dtype=DTYPE)
    vec[i, 0] = 1
    return vec


def bra(i, dim):
    """Get the numerical row vector.

    Args:
        i: Value of ket.
        dim: Dimension of qudits.
    """
    vec = torch.zeros((1, dim), dtype=DTYPE)
    vec[0, i] = 1
    return vec


def get_complex_tuple(mat, shape=None):
    """Convert the input matrix to a tuple which represents the complex matrix.

    Args:
        mat: Input matrix, which can be real or complex, and data type can be numpy.ndarray or
            torch.Tensor.
        shape: If not None, reshape the `mat` shape.
    """
    if isinstance(mat, (Tuple, List)):
        assert len(mat) == 2, "The input is real and imaginary part respectively."
        re, im = mat
        if isinstance(re, np.ndarray):
            re = torch.tensor(re, dtype=DTYPE)
            im = torch.tensor(im, dtype=DTYPE)
        elif isinstance(re, Tensor):
            re = re.clone().detach().type(DTYPE)
            im = im.clone().detach().type(DTYPE)
    elif isinstance(mat, np.ndarray):
        if np.iscomplexobj(mat):
            re = torch.tensor(mat.real, dtype=DTYPE)
            im = torch.tensor(mat.imag, dtype=DTYPE)
        else:
            re = torch.tensor(mat, dtype=DTYPE)
            im = torch.zeros_like(re)
    elif isinstance(mat, Tensor):
        if torch.is_complex(mat):
            re = mat.real.clone().detach().type(DTYPE)
            im = mat.imag.clone().detach().type(DTYPE)
        else:
            re = mat.clone().detach().type(DTYPE)
            im = torch.zeros_like(re)
    if shape:
        re = re.reshape(shape)
        im = im.reshape(shape)
    return re, im



def get_multi_value_controlled_gate_cmatrix(u_list: List[Tensor]):
    """Get the matrix of multi-value controlled gate by given matrix list.
    """
    assert len(u_list) == u_list[0].shape[0] == u_list[0].shape[1]
    dim = len(u_list)
    re = torch.zeros((dim**2, dim**2))
    im = torch.zeros((dim**2, dim**2))
    for i, u in enumerate(u_list):
        if isinstance(u, np.ndarray):
            u = torch.tensor(u)
        if torch.is_complex(u):
            re[i*dim: (i+1)*dim, i*dim: (i+1)*dim] = u.real
            im[i*dim: (i+1)*dim, i*dim: (i+1)*dim] = u.imag
        else:
            re[i*dim: (i+1)*dim, i*dim: (i+1)*dim] = u
    return re, im


def get_general_controlled_gate_cmatrix(u, dim: int, k_qudits=2, target_index=-1):
    """Get the matrix of general controlled gate.
    Args:
        u: Gate matrix.
        index_ctrl: The controlled state.
        k_qudits: The number of qudits the `u` acts on.
        target_index: It determines where to put the input matrix `u`.
    """
    re = torch.eye(dim**k_qudits)
    im = torch.zeros((dim**k_qudits, dim**k_qudits))
    if target_index < 0:
        target_index = 2**(k_qudits-1) - 1
    idx = dim * target_index
    u_re, u_im = get_complex_tuple(u)
    re[idx:idx+dim, idx:idx+dim] = u_re
    im[idx:idx+dim, idx:idx+dim] = u_im
    return re, im


def get_sigma_x_gate_cmatrix(i: int, j: int, dim: int, k_qudits: int = 1) -> Tuple:
    """Get the matrix of $\sigma_x$ gate. Read the documents of this package for more information.
    """
    re = ket(i, dim) * bra(j, dim) + ket(j, dim) * bra(i, dim)
    im = torch.zeros((dim, dim), dtype=DTYPE)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_sigma_y_gate_cmatrix(i: int, j: int, dim: int, k_qudits: int = 1) -> Tuple:
    """Get the matrix of $\sigma_y$ gate. Read the documents of this package for more information.
    """
    re = torch.zeros((dim, dim), dtype=DTYPE)
    im = -ket(i, dim) * bra(j, dim) + ket(j, dim) * bra(i, dim)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_sigma_z_gate_cmatrix(i: int, j: int, dim: int, k_qudits: int = 1) -> Tuple:
    """Get the matrix of $\sigma_z$ gate. Read the documents of this package for more information.
    """
    re = ket(i, dim) * bra(i, dim) - ket(j, dim) * bra(j, dim)
    im = torch.zeros((dim, dim), dtype=DTYPE)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_pauli_x_gate_cmatrix(i: int, j: int, dim: int, k_qudits: int = 1) -> Tuple:
    """Get the matrix of extended pauli-X gate. Read the documents of this package for more information.
    """
    re, im = get_sigma_x_gate_cmatrix(i, j, dim, k_qudits=1)
    for k in range(dim):
        if k != i and k != j:
            re += ket(k, dim) * bra(k, dim)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_pauli_y_gate_cmatrix(i: int, j: int, dim: int, k_qudits: int = 1) -> Tuple:
    """Get the matrix of extended pauli-Y gate. Read the documents of this package for more information.
    """
    re, im = get_sigma_y_gate_cmatrix(i, j, dim, k_qudits=1)
    for k in range(dim):
        if k != i and k != j:
            re += ket(k, dim) * bra(k, dim)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_pauli_z_gate_cmatrix(i: int, j: int, dim: int, k_qudits: int = 1) -> Tuple:
    """Get the matrix of extended pauli-Z gate. Read the documents of this package for more information.
    """
    re, im = get_sigma_z_gate_cmatrix(i, j, dim, k_qudits=1)
    for k in range(dim):
        if k != i and k != j:
            re += ket(k, dim) * bra(k, dim)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_rotate_x_gate_cmatrix(i: int, j: int, dim: int, pr: Parameter, k_qudits: int = 1) -> Tuple:
    """Get the matrix of rotate-X gate. Read the documents of this package for more information.
    """
    re = torch.eye(dim, dtype=DTYPE)
    im = torch.zeros((dim, dim), dtype=DTYPE)
    re[i, i] = torch.cos(pr / 2.)
    re[j, j] = torch.cos(pr / 2.)
    im[i, j] = -torch.sin(pr / 2.)
    im[j, i] = -torch.sin(pr / 2.)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_rotate_y_gate_cmatrix(i: int, j: int, dim: int, pr: Parameter, k_qudits: int = 1) -> Tuple:
    """Get the matrix of rotate-Y gate. Read the documents of this package for more information.
    """
    re = torch.eye(dim, dtype=DTYPE)
    im = torch.zeros((dim, dim), dtype=DTYPE)
    re[i, i] = torch.cos(pr / 2.)
    re[i, j] = -torch.sin(pr / 2.)
    re[j, i] = torch.sin(pr / 2.)
    re[j, j] = torch.cos(pr / 2.)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_rotate_z_gate_cmatrix(i: int, j: int, dim: int, pr: Parameter, k_qudits: int = 1) -> Tuple:
    """Get the matrix of rotate-Z gate. Read the documents of this package for more information.
    """
    re = torch.eye(dim, dtype=DTYPE)
    im = torch.zeros((dim, dim), dtype=DTYPE)
    re[i, i] = torch.cos(pr / 2.)
    re[j, j] = torch.cos(pr / 2.)
    im[i, i] = -torch.sin(pr / 2.)
    im[j, j] = torch.sin(pr / 2.)
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_increment_gate_cmatrix(dim: int, k_qudits: int = 1):
    """Get the matrix of increment gate. Read the documents of this package for more information.
    """
    re = torch.tensor(np.eye(dim, k=-1), dtype=DTYPE)
    re[dim-1, dim-1] = 1.0
    im = torch.zeros((dim, dim))
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_hadamard_gate_cmatrix(dim: int, k_qudits: int = 1):
    """Get the matrix of Hadamard gate. Read the documents of this package for more information.
    """
    re = torch.zeros((dim, dim), dtype=DTYPE)
    im = torch.zeros((dim, dim), dtype=DTYPE)
    for i in range(dim):
        for j in range(dim):
            re[i, j] = torch.cos(torch.tensor(2.0 * torch.pi * i * j / dim))
            im[i, j] = torch.sin(torch.tensor(2.0 * torch.pi * i * j / dim))
    if k_qudits >= 2:
        re, im = get_general_controlled_gate_cmatrix((re, im), dim, k_qudits)
    return re, im


def get_controlled_increment_gate_cmatrix(dim: int):
    """Get the matrix of controlled increment gate. Read the documents of this package for more
    information.
    """
    u = get_increment_gate_cmatrix(dim)
    re, im = get_general_controlled_gate_cmatrix(u, dim=dim, target_index=dim-1)
    return re, im


def get_single_controlled_gate_cmatrix(u, dim: int):
    """Get the matrix of single-controlled increment gate. Read the documents of this package for
    more information.
    """
    re, im = get_general_controlled_gate_cmatrix(u, dim=dim, target_index=dim-1)
    return re, im


def get_multi_controlled_gate_cmatrix(u, dim: int, k_qudits: int):
    """Get the matrix of multi-controlled increment gate. Read the documents of this package for
    more information.
    """
    re, im = get_general_controlled_gate_cmatrix(u, dim=dim, k_qudits=k_qudits, target_index=dim-1)
    return re, im


def evolution(op_mat: Tensor, qs: Tensor, target_indices: List[int]) -> Tensor:
    """Get the new quantum state after applying specific operation(gate or matrix).

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


class GateBase(nn.Module):
    """Base class for qudit gates.
    """

    def __init__(self, dim, name, obj_qudits=None, ctrl_qudits=None):
        """Initialize a QuditGate.
        """
        super().__init__()
        if not isinstance(name, str):
            raise TypeError(f'Excepted string for gate name, get {type(name)}')
        self.dim = dim
        self.name = name
        self.obj_qudits = obj_qudits
        self.ctrl_qudits = ctrl_qudits
        if obj_qudits:
            self.on(obj_qudits, ctrl_qudits)

    def on(self, obj_qudits, ctrl_qudits=None):
        """Define which qudits the gate act on and control qudits.
        """
        if isinstance(obj_qudits, int):
            obj_qudits = [obj_qudits]
        if isinstance(ctrl_qudits, int):
            ctrl_qudits = [ctrl_qudits]
        if ctrl_qudits is None:
            ctrl_qudits = []
        if obj_qudits is None:
            raise ValueError("The `obj_qudits` can't be None")
        if set(obj_qudits) & set(ctrl_qudits):
            raise ValueError(f'{self.name} obj_qudits {obj_qudits} and ctrl_qudits {ctrl_qudits} cannot be same')
        if len(set(obj_qudits)) != len(obj_qudits):
            raise ValueError(f'{self.name} gate obj_qudits {obj_qudits} cannot be repeated')
        if len(set(ctrl_qudits)) != len(ctrl_qudits):
            raise ValueError(f'{self.name} gate ctrl_qudits {ctrl_qudits} cannot be repeated')
        self.obj_qudits = obj_qudits
        self.ctrl_qudits = ctrl_qudits
        return self

    def forward(self, qs):
        """Get next state after the gate operation.
        """
        target_indices = self.ctrl_qudits + self.obj_qudits
        shape = (self.dim, self.dim) * len(target_indices)
        cmat = self._cmatrix()
        cmat = (cmat[0].reshape(shape), cmat[1].reshape(shape))
        return evolution_complex(cmat, qs, target_indices)

    def _cmatrix(self):
        """Get the gate matrix that in Tuple format.
        """
        return NotImplemented

    @property
    def matrix(self):
        """Get the matrix of gate.
        """
        cmat = self._cmatrix()
        return torch.complex(cmat[0], cmat[1])

    def is_unitary(self):
        """Check if this gate is unitary.
        """
        u = self.matrix()
        v = torch.matmul(u, u.conj().T)
        return torch.isclose(v.real, torch.eye(self.dim)).all() and \
            torch.isclose(v.imag, torch.zeros((self.dim, self.dim)))


class NoneParamGate(GateBase):
    """None Parameter Gate.
    """

    def __init__(self, dim, name, obj_qudits=None, ctrl_qudits=None):
        """Initialize an `NoneParamGate`.

        Args:
            dim: The dimension of qudits.
            name: The gate name.
            n_qudits: The number of qudits.
            obj_qudits: The objective qudits.
            ctrl_qudits: The control qudits.
        """
        super().__init__(dim, name, obj_qudits, ctrl_qudits)

    def __repr__(self):
        """Return a string representation of the object.
        """
        assert self.obj_qudits is not None, "There's no object qudit."

        str_obj = ' '.join(str(i) for i in self.obj_qudits)
        str_ctrl = ' '.join(str(i) for i in self.ctrl_qudits)
        if str_ctrl:
            return f'{self.name}({self.dim}|{str_obj} <-: {str_ctrl})'
        else:
            return f'{self.name}({self.dim}|{str_obj})'


class WithParamGate(GateBase):
    """Rotation qudit gate.
    """
    def __init__(self, dim, pr, name="WithParamGate", obj_qudits=None, ctrl_qudits=None):
        """Initialize an RotationGate.
        """
        super().__init__(dim, name, obj_qudits, ctrl_qudits)
        self.trainable = True
        self._build_parameter(pr)

    def _build_parameter(self, pr):
        """Construct the parameter according the input.
        """
        value = DEFAULT_VALUE
        name = DEFAULT_PARAM_NAME

        if pr is None:
            pass
        elif isinstance(pr, Parameter):
            self.param = pr
        elif isinstance(pr, str):
            value = 1.0
            name = pr
        elif isinstance(pr, (int, float, Tensor)):
            value = pr
        else:
            raise TypeError(f"Parameter `pr` should be str of Tensor(float32), but get {pr}")
        self.param = Parameter(Tensor([value]))
        self.param_name = name

    def __repr__(self):
        """Return a string representation of the object.
        """
        str_obj = ' '.join(str(i) for i in self.obj_qudits)
        str_ctrl = ' '.join(str(i) for i in self.ctrl_qudits)
        str_pr = self.param_name
        if str_ctrl:
            return f'{self.name}({self.dim} {str_pr}|{str_obj} <-: {str_ctrl})'
        else:
            return f'{self.name}({self.dim} {str_pr}|{str_obj})'

    def no_grad_(self):
        """No gradient for the parameter.
        """
        self.trainable = False
        if isinstance(self.param, Parameter):
            self.param.requires_grad = False

    def with_grad_(self):
        """Need gradient for the parameter.
        """
        self.trainable = True
        if isinstance(self.param, Parameter):
            self.param.requires_grad = True

    def assign_param(self, value):
        """Set the value of the parameter. Note: this function won't check if the shape
        of input is reasonable.
        """
        assert self.param is not None, "`param` is None."
        if isinstance(value, float):
            value = [value]
        self.param.data = Tensor(value)


class PauliNoneParamGate(NoneParamGate):
    """Pauli based none parameter gate. This gate contains two indexes.
    """
    def __init__(self, dim, i, j, name, obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, name, obj_qudits, ctrl_qudits)
        assert i < dim and j < dim, "The `i` and `j` should less than `dim`."
        self._i = i
        self._j = j

    def __repr__(self):
        str_obj = ' '.join(str(i) for i in self.obj_qudits)
        str_ctrl = ' '.join(str(i) for i in self.ctrl_qudits)
        if str_ctrl:
            return f'{self.name}({self.dim} {self._i} {self._j}|{str_obj} <-: {str_ctrl})'
        return f'{self.name}({self.dim} {self._i} {self._j}|{str_obj})'


class SX(PauliNoneParamGate):
    """Sigma-X gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1,
                 name="SX", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_sigma_x_gate_cmatrix(self._i, self._j, self.dim, k_qudits)



class SY(PauliNoneParamGate):
    """Sigma-Y gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1,
                 name="SY", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_sigma_y_gate_cmatrix(self._i, self._j, self.dim, k_qudits)


class SZ(PauliNoneParamGate):
    """Sigma-Z gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1,
                 name="SZ", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, name, obj_qudits, ctrl_qudits)


    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_sigma_z_gate_cmatrix(self._i, self._j, self.dim, k_qudits)


class X(PauliNoneParamGate):
    """Extended Pauli-X gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1,
                 name="X", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_pauli_x_gate_cmatrix(self._i, self._j, self.dim, k_qudits)


class Y(PauliNoneParamGate):
    """Extended Pauli-Y gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1,
                 name="Y", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_pauli_y_gate_cmatrix(self._i, self._j, self.dim, k_qudits)


class Z(PauliNoneParamGate):
    """Extended Pauli-Z gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1,
                 name="Z", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_pauli_z_gate_cmatrix(self._i, self._j, self.dim, k_qudits)


class H(NoneParamGate):
    """Hardarm gate for qudit.
    """
    def __init__(self, dim: int, name="H", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_hadamard_gate_cmatrix(self.dim, k_qudits)


class PauliWithParamGate(WithParamGate):
    """Pauli based none parameter gate. This gate contains two indexes.
    """
    def __init__(self, dim, i, j, pr, name, obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, pr, name, obj_qudits, ctrl_qudits)
        assert i < dim and j < dim, "The `i` and `j` should less than `dim`."
        self._i = i
        self._j = j

    def __repr__(self):
        str_obj = ' '.join(str(i) for i in self.obj_qudits)
        str_ctrl = ' '.join(str(i) for i in self.ctrl_qudits)
        if str_ctrl:
            return f'{self.name}({self.dim} {self._i} {self._j} {self.param_name}|{str_obj} <-: {str_ctrl})'
        else:
            return f'{self.name}({self.dim} {self._i} {self._j} {self.param_name}|{str_obj})'


class RX(PauliWithParamGate):
    """Rotation X gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1, pr=None,
                 name="RX", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, pr, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_rotate_x_gate_cmatrix(self._i, self._j, self.dim, self.param, k_qudits)


class RY(PauliWithParamGate):
    """Rotation Y gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1, pr=None,
                 name="RY", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, pr, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_rotate_y_gate_cmatrix(self._i, self._j, self.dim, self.param, k_qudits)


class RZ(PauliWithParamGate):
    """Rotation Z gate for qudit.
    """
    def __init__(self, dim: int = 2, i: int = 0, j: int = 1, pr=None,
                 name="RZ", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, i, j, pr, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_rotate_z_gate_cmatrix(self._i, self._j, self.dim, self.param, k_qudits)


class INC(NoneParamGate):
    """Increment gate for qudit.
    """
    def __init__(self, dim: int, name="INC", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        k_qudits = len(self.obj_qudits) + len(self.ctrl_qudits)
        return get_increment_gate_cmatrix(self.dim, k_qudits)


class MVCG(NoneParamGate):
    """Multi-value controlled gate.
    """
    def __init__(self, dim: int, u_list: List, name="MVCG", obj_qudits=None, ctrl_qudits=None):
        assert len(u_list) == dim, "The length of gates should equals to the `dim`."
        super().__init__(dim, name, obj_qudits, ctrl_qudits)
        self.u_list = u_list

    def _cmatrix(self):
        return get_multi_value_controlled_gate_cmatrix(self.u_list)


class CINC(NoneParamGate):
    """Controlled increment gate.
    """
    def __init__(self, dim: int, name="CINC", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, name, obj_qudits, ctrl_qudits)

    def _cmatrix(self):
        return get_controlled_increment_gate_cmatrix(self.dim)


class SCG(NoneParamGate):
    """Single controlled gate.
    """
    def __init__(self, dim: int, u, name="SCG", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, name, obj_qudits, ctrl_qudits)
        self.u = u

    def _cmatrix(self):
        return get_single_controlled_gate_cmatrix(self.u, self.dim)


class MCG(NoneParamGate):
    """Multi controlled gate for qudit.
    """
    def __init__(self, dim: int, u, k_qudits=2, name="MCG", obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, name, obj_qudits, ctrl_qudits)
        self.u = u
        self.k_qudits = k_qudits

    def _cmatrix(self):
        return get_general_controlled_gate_cmatrix(self.u, self.dim, self.k_qudits,
                                                   target_index=self.dim-1)


class UMG(NoneParamGate):
    """Universal math gate.
    """
    def __init__(self, dim: int, mat: Union[Tensor, Tuple], name="UMG",
                 obj_qudits=None, ctrl_qudits=None):
        super().__init__(dim, name, obj_qudits, ctrl_qudits)
        self._mat = mat

    def _cmatrix(self):
        return get_complex_tuple(self._mat)
