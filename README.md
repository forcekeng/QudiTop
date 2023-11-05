# QudiTop

This is a quantum simulator for qudit system based on the deep learning framework - PyTorch[1].

## Get Started

```python
from quditop.gates import H, X, Y, Z, RX, RY, RZ, UMG
from quditop.circuit import Circuit

dim = 2
n_qudits = 4

circ = Circuit(dim, n_qudits, gates=[
    H(dim).on(0),
    X(dim, [0,1]).on(0, [2,3], [1, 1]),
    Y(dim).on(1),
    Z(dim).on(1, 2, 1),
    RX(dim, pr='param').on(3),
    RY(dim, pr=1.0).on(2, 3, 1),
    X(dim).on(1, [0, 2, 3], 1),
    RY(dim, pr=2.0).on(3, [0, 1, 2], 1)
])

print(circ)
print(f"quantum state:\n{circ.get_qs(ket=True)}")
```

You can see more example in detail in the folder of `examples/`.

## File Structure

The core source code lies in `quditop/` folder and examples in `examples/`.

```log
.
├── README.md
├── examples/
│   ├── demo_basic.ipynb    # Demonstrate basic functions
│   └── demo_vqe.ipynb      # Demonstrate the VQE application
└── quditop/
    ├── circuit.py          # Define the Circuit class
    ├── expectation.py      # Define expectation
    ├── gates.py            # Define various gate
    ├── global_var.py       # Define some global configuration
    └── utils.py            # Some tool functions.
```

## Thanks to

[1] [pytorch/pytorch: Tensors and Dynamic neural networks in Python with strong GPU acceleration (github.com)](https://github.com/pytorch/pytorch)

[2] [pyquil.simulation._numpy — pyQuil 4.0.3 documentation (rigetti.com)](https://pyquil-docs.rigetti.com/en/stable/_modules/pyquil/simulation/_numpy.html)

[3] [【开发者群英会】在MindQuantum中实现任意维度的通用qudit量子线路模拟 · Issue #I7Q1UV · MindSpore/community - Gitee.com](https://gitee.com/mindspore/community/issues/I7Q1UV)

[4] [QuditVQE/QuditSim at main · GhostArtyom/QuditVQE (github.com)](https://github.com/GhostArtyom/QuditVQE/tree/main/QuditSim)
