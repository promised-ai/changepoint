# pychangepoint

Python bindings for important functionality of changepoint

## Installation

Requires the [maturin](https://github.com/PyO3/maturin) build tool.

```
$ pip install maturin
```

To install the package, navigate to the `changepoint-py` directory and run

```
$ maturin develop
```

## Example

```python
import pychangepoint as pycpt

ng = pycpt.Ng(0.0, 1.0, 1.0, 1.0);
bocpd = pycpt.BocpdNg(50, ng);

data = [
    1.0, 2.0, 1.0, 2.0, 1.5, 1.5, 1.0, 2.0,
    10.0, 11.0, 9.0, 10.0, 9.0, 11.0
]

rs = [bocpd.step(datum) for datum in data]

pycpt.infer_changepoints(rs, 1000)
# [
#     1.0,
#     0.001,
#     0.004,
#     0.001,
#     0.001,
#     0.009,
#     0.081,
#     0.938,
#     0.002,
#     0.0,
#     0.0,
#     0.0,
#     0.001,
#     0.026
# ]
```
