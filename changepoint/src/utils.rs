//! General Utilities

use rand::Rng;
use rv::{
    misc::argmax, prelude::Categorical, prelude::CategoricalError, prelude::Rv,
};
use std::fmt::Display;
use std::fs::File;
use std::io::{self, prelude::*};

/// Writes data and R to `{prefix}_data.txt` and `{prefix}_r.txt`, respectively.
///
/// # Errors
/// If data cannot be written to disk, an error is returned.
#[allow(clippy::ptr_arg)]
pub fn write_data_and_r<T: Display>(
    prefix: &str,
    data: &[T],
    r: &Vec<Vec<f64>>,
    change_points: &[usize],
) -> io::Result<()> {
    // Write Data
    let data_file_path = format!("{prefix}_data.txt");
    let mut data_f = File::create(data_file_path)?;
    data.iter().try_for_each(|d| writeln!(data_f, "{d}"))?;

    // Write R
    let r_file_path = format!("{prefix}_r.txt");
    let mut r_f = File::create(r_file_path)?;

    let t = r.len();
    r.iter()
        .enumerate()
        .try_for_each::<_, io::Result<()>>(|(i, rs)| {
            for r in rs {
                write!(r_f, "{r},")?;
                if r.is_nan() || r.is_infinite() {
                    eprintln!("NaN/Infinite value in output: Row {i}");
                }
            }
            for _ in rs.len()..t {
                write!(r_f, ",")?;
            }
            writeln!(r_f)?;
            Ok(())
        })?;

    // Write change points
    let change_points_path = format!("{prefix}_change_points.txt");
    let mut cp_f = File::create(change_points_path)?;
    change_points
        .iter()
        .try_for_each(|cp| writeln!(cp_f, "{cp}"))?;

    Ok(())
}

/// Compute the locations in a sequence of distributions where the total
/// probability from 1 to window is above threshold.
#[allow(clippy::ptr_arg)]
#[must_use]
pub fn window_over_threshold(
    r: &Vec<Vec<f64>>,
    window: usize,
    threshold: f64,
) -> Vec<usize> {
    let mut result = Vec::new();
    for (i, rs) in r.iter().enumerate() {
        let window_sum: f64 = rs.iter().skip(1).take(window).sum();
        if window_sum >= threshold {
            result.push(i);
        }
    }
    result
}

/// Infer change-point locations from the run-length distributions
///
/// This method works by walking backwards through the run-length distributions jumping back to
/// past distributions until the process ends at the first step.
///
/// # Parameters
/// * `rs` - Run lengths probability distributions for each step observed
/// * `sample_size` - Number of monte-carlo draws from the rl distribution
/// * `rng` - Random number generator
///
/// # Returns
/// The return value is the proportion of samples which show a change-point at the given index.
///
/// # Errors
/// If the values in `rs` are cannot be converted to categorical weights, this will return an
/// error.
pub fn infer_changepoints<R: Rng>(
    rs: &[Vec<f64>],
    sample_size: usize,
    rng: &mut R,
) -> Result<Vec<f64>, CategoricalError> {
    let n = rs.len();
    let dists: Vec<Categorical> =
        rs.iter()
            .map(|r| Categorical::new(r))
            .collect::<Result<Vec<Categorical>, CategoricalError>>()?;

    let counts: Vec<usize> =
        (0..sample_size).fold(vec![0_usize; n], |mut acc, _| {
            let mut s: usize = n - 1;
            while s != 0 {
                let cur_run_length: usize = dists[s].draw(rng);
                s = s.saturating_sub(cur_run_length);
                acc[s] += 1;
            }
            acc
        });

    Ok(counts
        .into_iter()
        .map(|c| (c as f64) / (sample_size as f64))
        .collect())
}

/// Creates a pseudo cmf distribution for change-point locations.
///
/// This calculates the cumulative sum of the `infer_changepoints` return value mod 1.0.
///
/// See [`infer_changepoints`](fn.infer_changepoints.html) for more detail.
///
/// # Errors
/// If `infer_pseudo_cmf_changepoints` returns an error, this will as well.
pub fn infer_pseudo_cmf_changepoints<R: Rng>(
    rs: &[Vec<f64>],
    sample_size: usize,
    rng: &mut R,
) -> Result<Vec<f64>, CategoricalError> {
    let ps = infer_changepoints(rs, sample_size, rng)?;
    Ok(ps
        .into_iter()
        .scan(0.0, |acc, x| {
            *acc = (*acc + x).rem_euclid(1.0);
            Some(*acc)
        })
        .collect())
}

//    0 1 2 3 4 5 6 7
// 0 [1]                  | s = 0 DONE
// 1 [0 1]                |
// 2 [0 0 1]              |
// 3 [0 0 0 1]            | s = 3, argmax = 3, s -> s - 3 = 0, CPs: [7, 4, 0]
// 4 [1] - Change         | s = 4, argmax = 0, s -> s - 1 = 3, CPs: [7, 4]
// 5 [0 1]                |
// 6 [0 0 1]              | s = 6, argmax = 2, s -> s - 2 = 4, CPs: [7, 4]
// 7 [1] - Change         | s = 7, argmax = 0, s -> s - 1 = 6, CPs: [7]
// 8 [0 1]                | s = 8, argmax = 1, s -> s - 1 = 7, CPs: [7]

/// Maximum a posteriori change points
///
/// This reverse walks through the run-length distribution sequence and only takes the most likely
/// set of change-points.
#[must_use]
pub fn map_changepoints(r: &[Vec<f64>]) -> Vec<usize> {
    let mut s = r.len() - 1;
    let mut change_points: Vec<usize> = vec![];
    while s != 0 {
        let most_likely_runlength: usize =
            *argmax(&r[s]).first().expect("r should not be empty");

        if most_likely_runlength == 0 {
            if let Some(last) = change_points.last() {
                if *last != s {
                    change_points.push(s);
                }
            }
            s = s.saturating_sub(1);
        } else {
            s = s.saturating_sub(most_likely_runlength);
            change_points.push(s);
        }
    }
    change_points.reverse();
    change_points
}

fn diff<T>(a: &T, b: &T) -> T
where
    T: PartialOrd + std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>,
{
    if a > b {
        a - b
    } else {
        b - a
    }
}

/// The max-norm or max-error between two sequences.
///
/// # Panics
/// If the input slices are not of equal length, this will panic.
pub fn max_error<T>(predicted: &[T], expected: &[T]) -> T
where
    T: PartialOrd + std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>,
{
    debug_assert_eq!(
        predicted.len(),
        expected.len(),
        "predicted and expected must be the same size."
    );
    assert!(!predicted.is_empty(), "Sequences cannot be empty");
    predicted.iter().skip(1).zip(expected.iter().skip(1)).fold(
        diff(predicted.first().unwrap(), expected.first().unwrap()),
        |acc, (a, b)| {
            let d: T = diff(a, b);
            match acc.partial_cmp(&d).unwrap() {
                std::cmp::Ordering::Less | std::cmp::Ordering::Equal => acc,
                std::cmp::Ordering::Greater => d,
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::map_changepoints;

    #[test]
    fn simple_map() {
        let r: [Vec<f64>; 9] = [
            vec![1.],
            vec![0., 1.],
            vec![0., 0., 1.],
            vec![0., 0., 0., 1.],
            vec![1.],
            vec![0., 1.],
            vec![0., 0., 1.],
            vec![1.],
            vec![0., 1.],
        ];

        assert_eq!(map_changepoints(&r), vec![0, 4, 7]);
    }

    #[test]
    fn consecutive_map() {
        let r: [Vec<f64>; 9] = [
            vec![1.],
            vec![0., 1.],
            vec![0., 0., 1.],
            vec![0., 0., 0., 1.],
            vec![1.],
            vec![1.],
            vec![1.],
            vec![1.],
            vec![0., 1.],
        ];

        assert_eq!(map_changepoints(&r), vec![0, 4, 5, 6, 7]);
    }
}
