use cpd::*;
use criterion::*;
use rv::prelude::*;
use std::sync::Arc;

fn bench_online_bayesian(c: &mut Criterion) {
    let raw_data: &str = include_str!("../resources/spx.csv");
    let data: Vec<f64> = raw_data
        .lines()
        .skip(1)
        .map(|line| line.split_at(10).1.parse().unwrap())
        .collect();

    c.bench(
        "online bayesian",
        ParameterizedBenchmark::new(
            "process",
            move |b, nelems| {
                b.iter(|| {
                    // Create the BOCPD processor
                    let mut cpd = BOCPD::new(
                        constant_hazard(250.0),
                        &Gaussian::standard(),
                        Arc::new(NormalGamma::new(0.0, 1.0, 1.0, 1E-5).unwrap()),
                    );

                    // Feed data into change point detector
                    let _res: Vec<Vec<f64>> = data
                        .clone()
                        .iter()
                        .take(*nelems)
                        .map(|d| cpd.step(d))
                        .collect();
                })
            },
            (0..500).step_by(100),
        )
        .throughput(|&nelems| Throughput::Elements(nelems as u32)),
    );
}

criterion_group!(benches, bench_online_bayesian);
criterion_main!(benches);
