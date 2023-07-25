def test_int_to_bool():
    """Test for issue #5 by HilaManor

    The failure is the NumPy types `int64` and `bool_` aren't supported by
    default in pyo3.
    """
    import changepoint as cpt
    import numpy as np

    cpd = cpt.Bocpd(
        prior=cpt.BetaBernoulli(),
        lam=30,
    )

    bers1 = np.random.binomial(1, 0.1, size=10)
    bers2 = np.random.binomial(1, 0.2, size=10)
    bers3 = np.random.binomial(1, 0.3, size=10)
    bers4 = np.random.binomial(1, 0.1, size=10)

    data = np.concatenate([bers1, bers2, bers3, bers4])

    n = len(data)
    change_point_history = np.zeros((n, n))
    for i, x in enumerate(data):
        change_point_history[i, : i + 1] = cpd.step(x)
