//! General Utilities

use rv::misc::argmax;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Writes data and R to `{prefix}_data.txt` and `{prefix}_r.txt`, respectively.
#[allow(clippy::ptr_arg)]
pub fn write_data_and_r<T: Display>(
    prefix: &str,
    data: &[T],
    r: &Vec<Vec<f64>>,
    change_points: &[usize],
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
        .enumerate()
        .map(|(i, rs)| {
            for r in rs {
                write!(r_f, "{},", r)?;
                if r.is_nan() || r.is_infinite() {
                    eprintln!("NaN/Infinite value in output: Row {}", i);
                }
            }
            for _ in rs.len()..t {
                write!(r_f, ",")?;
            }
            writeln!(r_f)?;
            Ok(())
        })
        .collect::<io::Result<()>>()?;

    // Write change points
    let change_points_path = format!("{}_change_points.txt", prefix);
    let mut cp_f = File::create(change_points_path)?;
    change_points
        .iter()
        .map(|cp| writeln!(cp_f, "{}", cp))
        .collect::<io::Result<()>>()?;

    Ok(())
}

/// Compute the locations in a sequence of distributions where the total
/// probability from 1 to window is above threshold.
#[allow(clippy::ptr_arg)]
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

/// Alternative methods for detecting change points
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum ChangePointDetectionMethod {
    /// Detect when the most likely path is reset to zero.
    Reset,
    /// Detect when the most likely path is not incremented by one.
    NonIncremental,
    /// Detect when the most likely path's length drops by some fraction.
    DropThreshold(f64),
    /// Detect negative change in MAP run length
    NegativeChangeInMAP,
}

impl ChangePointDetectionMethod {
    /// Detect changepoints with the given method
    #[allow(clippy::needless_range_loop, clippy::ptr_arg)]
    pub fn detect(&self, r: &Vec<Vec<f64>>) -> Vec<usize> {
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
            Self::NegativeChangeInMAP => {
                let mut last_argmax = *argmax(&r[0]).last().unwrap();
                for i in 1..r.len() {
                    let this_argmax = *argmax(&r[i]).last().unwrap();
                    if this_argmax < last_argmax {
                        res.push(i);
                    }
                    last_argmax = this_argmax;
                }
            }
        }
        res
    }
}
