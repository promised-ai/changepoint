import changepoint as cpt


def test_map_changepoints():
    bocpd = cpt.Bocpd(cpt.NormalGamma(), 50)

    data = [
        1.0,
        2.0,
        1.0,
        2.0,
        1.5,
        1.5,
        1.0,
        2.0,
        10.0,
        11.0,
        9.0,
        10.0,
        9.0,
        11.0,
    ]

    rs = [bocpd.step(datum) for datum in data]

    assert cpt.map_changepoints(rs) == [0, 7]
