//! General Utilities

use rv::misc::argmax;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::io::prelude::*;

/// Writes data and R to `{prefix}_data.txt` and `{prefix}_r.txt`, respectively.
pub fn write_data_and_r<T: Display>(
    prefix: &str,
    data: &[T],
    r: &Vec<Vec<f64>>,
    change_points: &Vec<usize>,
) -> io::Result<()> {
    // Write Data
    let data_file_path = format!("{}_data.txt", prefix);
    let mut data_f = File::create(data_file_path)?;
    data.iter()
        .map(|d| writeln!(data_f, "{}", d))
        .collect::<io::Result<()>>()?;

    // Write R
    let r_file_path = format!("{}_r.txt", prefix);
    let mut r_f = File::create(r_file_path)?;

    let t = r.len();
    r.iter()
        .map(|rs| {
            for i in 0..rs.len() {
                write!(r_f, "{} ", rs[i])?;
            }
            for _ in rs.len()..t {
                write!(r_f, "{} ", 0.0)?;
            }
            writeln!(r_f, "")?;
            Ok(())
        })
        .collect::<io::Result<()>>()?;

    // Write change points
    let change_points_path = format!("{}_change_points.txt", prefix);
    let mut cp_f = File::create(change_points_path)?;
    change_points.iter().map(|cp| writeln!(cp_f, "{}", cp)).collect::<io::Result<()>>()?;

    Ok(())
}

/// Compute the locations in a sequence of distributions where the total
/// probability from 1 to window is above threshold.
pub fn window_over_threshold(r: &Vec<Vec<f64>>, window: usize, threshold: f64) -> Vec<usize> {
    let mut result = Vec::new();
    for (i, rs) in r.iter().enumerate() {
        let window_sum: f64 = rs.iter().skip(1).take(window).sum();
        if window_sum >= threshold {
            result.push(i);
        }
    }
    result
}

/// Alternative methods for detecting change points
pub enum ChangePointDetectionMethod {
    /// Detect when the most likely path is reset to zero.
    Reset,
    /// Detect when the most likely path is not incremented by one.
    NonIncremental,
    /// Detect when the most likely path's length drops by some fraction.
    DropThreshold(f64),
}

impl ChangePointDetectionMethod {
    fn detect(&self, r: Vec<Vec<f64>>) -> Vec<usize> {
        let mut res = Vec::new();
        match self {
            ChangePointDetectionMethod::Reset => {
                for i in 1..r.len() {
                    if argmax(&r[i]).iter().any(|&x| x == 0) {
                        res.push(i);
                    }
                }
            }
            ChangePointDetectionMethod::NonIncremental => {
                let mut last_argmax: usize = *argmax(&r[0]).last().unwrap();
                for i in 1..r.len() {
                    let this_argmax = *argmax(&r[i]).last().unwrap();
                    if last_argmax + 1 != this_argmax {
                        res.push(i);
                    }
                    last_argmax = this_argmax;
                }
            }
            ChangePointDetectionMethod::DropThreshold(t) => {
                let mut last_argmax: usize = *argmax(&r[0]).last().unwrap();
                for i in 1..r.len() {
                    let this_argmax = *argmax(&r[i]).last().unwrap();
                    if (this_argmax as f64) / (last_argmax as f64) < 1.0 - t {
                        res.push(i);
                    }
                    last_argmax = this_argmax;
                }
            }
        }
        res
    }
}

/// Return the points at which the sequence has most likely changed paths.
/// Algorithm from: https://youtu.be/cas__TaFk9U
pub fn most_likely_breaks(r: &Vec<Vec<f64>>, method: ChangePointDetectionMethod) -> Vec<usize> {
    // TODO: This can be made faster
    let mut log_r: Vec<Vec<f64>> = r
        .iter()
        .map(|vs| vs.iter().map(|x| x.ln()).collect())
        .collect();

    // Accumulate probabilities
    for i in 0..(log_r.len() - 1) {
        let mut max = std::f64::NEG_INFINITY;
        for j in 0..(log_r[i].len()) {
            log_r[i + 1][j + 1] += log_r[i][j];
            max = max.max(log_r[i][j]);
        }
        log_r[i + 1][0] += max;
    }

    // Determine where step backward, instead of forwards, occurs.
    method.detect(log_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn most_likely_breaks_explicite() {
        let data: Vec<Vec<f64>> = vec![
            vec![1.0],                               // 0
            vec![0.1, 0.9],                          // 1
            vec![0.9, 0.0, 0.1],                     // 2
            vec![0.1, 0.7, 0.0, 0.2],                // 3
            vec![0.2, 0.2, 0.3, 0.0, 0.3],           // 4
            vec![0.1, 0.4, 0.1, 0.1, 0.0, 0.3],      // 5
            vec![0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5], //6
        ];

        let breaks = most_likely_breaks(&data.to_vec(), ChangePointDetectionMethod::NonIncremental);
        assert_eq!(breaks, vec![2, 5, 6]);
    }

}
