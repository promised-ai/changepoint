import pychangepoint as pycpt

data = [
    1.0, 2.0, 1.0, 2.0, 1.5, 1.5, 1.0, 2.0,
    10.0, 11.0, 9.0, 10.0, 9.0, 11.0
]

ng = pycpt.Ng(0.0, 1.0, 1.0, 1.0);
bocpd = pycpt.BocpdNg(50, ng);
rs = [bocpd.step(datum) for datum in data]

print(pycpt.infer_changepoints(rs, 1000))

# --- 
kernel_args = pycpt.KernelArgs(0.5, 10.0, 0.1);
argpcpd = pycpt.ArgpCpd(kernel_args, 3, 2.0, 1.0, -5.0, 1.0, 1.0);
rs = [argpcpd.step(datum) for datum in data]

print(pycpt.infer_changepoints(rs, 1000))
