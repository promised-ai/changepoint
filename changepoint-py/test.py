import pychangepoint as pycpt


def print_box(xs):
    n = len(xs) * 6 + 1
    print('┌', end='')
    print('─' * n, end='')
    print('┐')

    print('│', end='')
    for x in xs:
        print("{:6.3f}".format(x), end='')
    print(' │')

    print('└', end='')
    print('─' * n, end='')
    print('┘')


data = [
    1.0, 2.0, 1.0, 2.0, 1.5, 1.5, 1.0, 2.0,
    10.0, 11.0, 9.0, 10.0, 9.0, 11.0
]

# BOCPD
# -----
bocpd = pycpt.BocpdNg(50);
rs = [bocpd.step(datum) for datum in data]

print_box(pycpt.infer_changepoints(rs, 1000))

# ARGP
# ----
argpcpd = pycpt.ArgpCpd();
rs = [argpcpd.step(datum) for datum in data]

print_box(pycpt.infer_changepoints(rs, 1000))
