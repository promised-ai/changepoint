use changepoint::*;
use criterion::*;
use rv::prelude::*;

fn bench_online_bayesian(c: &mut Criterion) {
    let raw_data: &str = include_str!("../../resources/TB3MS.csv");
    let data: Vec<f64> = raw_data
        .lines()
        .skip(1)
        .map(|line| line.split_at(11).1.parse().unwrap())
        .collect();

    let mut group = c.benchmark_group("Bocpd");
    for nelems in (0..500).step_by(100) {
        let subdata: Vec<f64> = data.iter().take(nelems).copied().collect();

        group.throughput(Throughput::Elements(nelems as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(nelems),
            &subdata,
            |b, data| {
                b.iter(|| {
                    // Create the Bocpd processor
                    let mut cpd = Bocpd::new(
                        250.0,
                        NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
                    );

                    // Feed data into change point detector
                    let _res: Vec<Vec<f64>> =
                        data.iter().map(|d| cpd.step(d).to_vec()).collect();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_online_bayesian);
criterion_main!(benches);
