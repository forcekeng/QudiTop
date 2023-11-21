"""Some global variables."""
import torch

DTYPE = torch.float64
CDTYPE = torch.complex128
DEFAULT_VALUE = torch.tensor(0, dtype=DTYPE)
DEFAULT_PARAM_NAME = '_param_'