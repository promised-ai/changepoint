# changepoint

Python bindings for important functionality of the rust library [changepoint](https://crates.io/crates/changepoint), a library for doing change point detection for steams of data.

## Installation

Install via pip with
```bash
$ python3 -m pip install "changepoint"
```

__Note__: If there is no binary distribution for your OS, architecture, and Python version, you will need the Rust compiler to build the package and install a Python tool called Maturin:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
python3 -m pip install maturin
```

See [rustup.rs](https://rustup.rs/) for instructions on installing Rust.

## Quick Docs
By convention in these docs and examples, 
```python
import changepoint as cpt
```

### Models
#### Bocpd

The Bayesian change point detector, `Bocpd`, takes a prior distribution, aka one of
```python,ignore
cpt.BetaBernoulli
cpt.NormalGamma
cpt.NormalInvChiSquared
cpt.NormalInvGamma
cpt.NormalInvWishart
cpt.PoissonGamma
```

Then, a `Bocpd` may be created:
```python
cpd = cpt.Bocpd(
    prior=cpt.NormalGamma(),
    lam=12,
)
```
where the prior is a `NormalGamma` and the characteristic run length is `12`.

Each step of the data stream, `data`, can be processed by
```python
import random
import numpy as np

data = [random.gauss() for _ in range(30)] \
    + [random.gauss(1, 2) for _ in range(30)]

n = len(data)
change_point_history = np.zeros((n, n))
for i, x in enumerate(data):
    change_point_history[i, : i + 1] = cpd.step(x)

print(cpt.map_changepoints(change_point_history))
```


#### ArgpCpd

`ArgpCpd` has an implicit prior as it is a Gaussian Process of the form `X_{i+1} = c X_{i-l-1, ..., i-1} + ε`
where `c` is the scale, `X_{i-l-1, ..., i-1}` is the previous vales in the sequence (`l` is the max-lag parameter), and `ε` is a white noise parameter.
It behaves similarity to the `Bocpd` class; for example,

```python
argp = cpt.ArgpCpd(logistic_hazard_h=-2, scale=3, noise_level=0.01)
n = len(data)
change_point_history = np.zeros((n + 1, n + 1))
xs = []
ys = []
for i, x in enumerate(data):
    cps = argp.step(x)
    change_point_history[i, : len(cps)] = cps

print(cpt.map_changepoints(change_point_history))
```

## Example

An example IPython notebook can be found [here](https://gitlab.com/Redpoll/changepoint/-/blob/master/changepoint-py/ChangePointExample.ipynb).
