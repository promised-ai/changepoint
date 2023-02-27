//! A set of toy models that produce deterministic changes

use changepoint::{utils, BocpdLike, BocpdTruncated};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rv::prelude::*;
use std::fs::File;
use std::{f64::NEG_INFINITY, io::prelude::*, iter::repeat};
use utils::infer_changepoints;

fn constant_sequence<'a, R: Rng>(
    c: f64,
    stddev: f64,
    rng: &'a mut R,
) -> impl Iterator<Item = f64> + 'a {
    let dist = Gaussian::new_unchecked(c, stddev);
    (0..).map(move |_| dist.draw(rng))
}

fn line_sequence<'a, R: Rng>(
    start: f64,
    delta: f64,
    stddev: f64,
    rng: &'a mut R,
) -> impl Iterator<Item = f64> + 'a {
    let noise = Gaussian::new_unchecked(0.0, stddev);

    (0..)
        .scan(start, move |state, _| {
            *state += delta;
            Some(*state)
        })
        .map(move |x| {
            let n: f64 = noise.draw(rng);
            x + n
        })
}

fn exp_sequence<'a, R: Rng>(
    start: f64,
    k: f64,
    stddev: f64,
    rng: &'a mut R,
) -> impl Iterator<Item = f64> + 'a {
    let noise = Gaussian::new_unchecked(0.0, stddev);
    (0..)
        .scan(start, move |state, _| {
            *state *= k + 1.0;
            Some(*state)
        })
        .map(move |x| {
            let n: f64 = noise.draw(rng);
            x + n
        })
}

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut rng = SmallRng::seed_from_u64(0xABCD);

    // Generate a sequence
    println!("Generating sequence");
    let mut seq: Vec<f64> =
        constant_sequence(10.0, 8.0, &mut rng).take(500).collect();

    seq.append(
        &mut (line_sequence(10.0, 10.0 / 500.0, 2.0, &mut rng)
            .take(500)
            .collect::<Vec<f64>>()),
    );
    seq.append(
        &mut (constant_sequence(10.0, 2.0, &mut rng)
            .take(500)
            .collect::<Vec<f64>>()),
    );

    seq.append(
        &mut (exp_sequence(10.0, 0.002, 2.0, &mut rng).take(500).collect()),
    );

    println!("Creating cpd");
    let mut cpd = BocpdTruncated::new(
        100.0,
        NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
    );

    cpd.preload(&seq.iter().take(20).cloned().collect::<Vec<f64>>());

    // Do run-length based change point detection
    println!("Generating run-length probabilities");
    let mut rs: Vec<Vec<f64>> = vec![];
    let mut posterior_samples: Vec<Vec<f64>> = vec![];
    for x in seq.iter() {
        let post = cpd.pp();
        let ps: Vec<f64> = (0..1000)
            .map(|_| post.draw(&mut rng).draw(&mut rng))
            .collect();
        posterior_samples.push(ps);
        rs.push(cpd.step(x).to_vec());
    }

    println!("Tracing runlengths");
    let change_point_probs = infer_changepoints(&rs, 5000, &mut rng)?;

    let mut f_seq = File::create("./toy_seq.txt")?;
    for x in seq {
        writeln!(f_seq, "{}", x)?;
    }

    let mut f_log_rs = File::create("./toy_rs.txt")?;
    let rs_len = rs.len();
    for r in rs {
        let line: String = r
            .iter()
            .chain(repeat(&NEG_INFINITY).take(rs_len - r.len()))
            .map(|x| x.to_string())
            .fold(String::new(), |acc, x| format!("{} {}", acc, x));
        writeln!(f_log_rs, "{}", line)?;
    }

    let mut f_change_point_probs = File::create("./toy_change_points.txt")?;
    for cp in change_point_probs {
        writeln!(f_change_point_probs, "{}", cp)?;
    }

    let mut f_ps = File::create("./toy_post_samples.txt")?;
    for ps in posterior_samples {
        let line: String = ps
            .iter()
            .map(|x| x.to_string())
            .fold(String::new(), |acc, x| format!("{} {}", acc, x));
        writeln!(f_ps, "{}", line)?;
    }

    Ok(())
}
