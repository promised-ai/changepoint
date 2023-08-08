import numpy as np
import pytest
import changepoint as cpt

# cpt.BetaBernoulli
# cpt.NormalGamma
# cpt.NormalInvChiSquared
# cpt.NormalInvGamma
# cpt.NormalInvWishart
# cpt.PoissonGamma

BINARY_SEQS = [
    [0, 1, 1, 0, 1, 1],
    [False, True, True, False, True, True],
    np.array([0, 1, 1, 0, 1, 1], dtype=np.int_),
    np.array([0, 1, 1, 0, 1, 1], dtype=np.intc),
    np.array([0, 1, 1, 0, 1, 1], dtype=np.uint),
    np.array([0, 1, 1, 0, 1, 1], dtype=np.uintc),
    np.array([False, True, True, False, True, True], dtype=np.bool_),
]


@pytest.mark.parametrize("seq", BINARY_SEQS)
def test_beta_bernoulli(seq):
    cpd = cpt.Bocpd(
        prior=cpt.BetaBernoulli(),
        lam=12,
    )
    for x in seq:
        cpd.step(x)


@pytest.mark.parametrize("dtype", [int, np.int_, np.uint, np.intc, np.uintc])
def test_beta_poission_gamma(dtype):
    seq = np.random.randint(10, size=(20,), dtype=dtype)
    cpd = cpt.Bocpd(
        prior=cpt.PoissonGamma(),
        lam=12,
    )
    for x in seq:
        cpd.step(x)


@pytest.mark.parametrize(
    "prior", [cpt.NormalGamma, cpt.NormalInvGamma, cpt.NormalInvChiSquared]
)
@pytest.mark.parametrize("dtype", [float, np.float_, np.single, np.double])
def test_normal_x(prior, dtype):
    seq = np.array(np.random.randn(20), dtype=dtype)
    cpd = cpt.Bocpd(
        prior=prior(),
        lam=12,
    )
    for x in seq:
        cpd.step(x)
