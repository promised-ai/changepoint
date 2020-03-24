use criterion::*;
use std::collections::VecDeque;

fn vec_push_front(n_start: usize, n_add: usize) {
    let mut vec = vec![0; n_start];
    for _ in 0..n_add {
        vec.insert(0, 0);
    }
}

fn deque_push_front(n_start: usize, n_add: usize) {
    let mut deque: VecDeque<usize> = vec![0; n_start].into();
    for _ in 0..n_add {
        deque.push_front(0);
    }
}


fn bench_vec_vs_deque(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_vs_deque");
    for i in [20usize, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("vec_push_front_1000", i),
            i,
            |b, i| b.iter(|| vec_push_front(1000, *i))
        );
        group.bench_with_input(
            BenchmarkId::new("deque_push_front_1000", i),
            i,
            |b, i| b.iter(|| deque_push_front(1000, *i))
        );
    }

    group.finish();
}

criterion_group!(benches, bench_vec_vs_deque);
criterion_main!(benches);
