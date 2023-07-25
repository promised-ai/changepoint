import changepoint as cpt


def print_box(xs):
    n = len(xs) * 6 + 1
    print("┌", end="")
    print("─" * n, end="")
    print("┐")

    print("│", end="")
    for x in xs:
        print("{:6.3f}".format(x), end="")
    print(" │")

    print("└", end="")
    print("─" * n, end="")
    print("┘")


def test_example():
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

    # BOCPD
    # -----
    bocpd = cpt.Bocpd(cpt.NormalGamma(), 50)
    rs = [bocpd.step(datum) for datum in data]

    print_box(cpt.infer_changepoints(rs, 1000))

    # ARGP
    # ----
    argpcpd = cpt.ArgpCpd()
    rs = [argpcpd.step(datum) for datum in data]

    print_box(cpt.infer_changepoints(rs, 1000))
