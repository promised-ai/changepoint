use changepoint::*;
use criterion::*;
use rv::process::gaussian::kernel::{ConstantKernel, RBFKernel, WhiteKernel};

fn bench_argpcpd(c: &mut Criterion) {
    let raw_data: &str = include_str!("../../resources/TB3MS.csv");
    let data: Vec<f64> = raw_data
        .lines()
        .skip(1)
        .map(|line| line.split_at(11).1.parse().unwrap())
        .collect();

    let mut group = c.benchmark_group("Argpcp");

    for nelems in (0..500).step_by(100) {
        let subdata: Vec<f64> = data.iter().take(nelems).copied().collect();

        group.throughput(Throughput::Elements(nelems as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(nelems),
            &subdata,
            |b, data| {
                b.iter(|| {
                    let constant_kernel = ConstantKernel::new(0.5).unwrap();
                    let rbf_kernel = RBFKernel::new(10.0).unwrap();
                    let white_kernel = WhiteKernel::new(0.01).unwrap();
                    let kernel = constant_kernel * rbf_kernel + white_kernel;
                    // Create the Argpcp processor
                    let mut cpd =
                        Argpcp::new(kernel, 3, 2.0, 1.0, -5.0, 1.0, 1.0);

                    // Feed data into change point detector
                    let _res: Vec<Vec<f64>> =
                        data.iter().map(|d| cpd.step(d).to_vec()).collect();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_argpcpd);
criterion_main!(benches);
