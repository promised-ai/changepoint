import pychangepoint as pycpt

ng = pycpt.Ng(0.0, 1.0, 1.0, 1.0);
bocpd = pycpt.BocpdNg(50, ng);

data = [
    1.0, 2.0, 1.0, 2.0, 1.5, 1.5, 1.0, 2.0,
    10.0, 11.0, 9.0, 10.0, 9.0, 11.0
]

rs = [bocpd.step(datum) for datum in data]

pycpt.infer_changepoints(rs, 1000)
