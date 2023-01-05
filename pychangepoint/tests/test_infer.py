import pychangepoint as pycpt


def test_map_changepoints():
    bocpd = pycpt.Bocpd(pycpt.NormalGamma(), 50)

    data = [
        1.0, 2.0, 1.0, 2.0, 1.5, 1.5, 1.0, 2.0,
        10.0, 11.0, 9.0, 10.0, 9.0, 11.0
    ]

    rs = [bocpd.step(datum) for datum in data]

    assert pycpt.map_changepoints(rs) == [0, 6]
