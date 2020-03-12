//! General Utilities

use crate::run_length_detector::{
    MapPathDetector, MapPathResult, RunLengthDetector,
};
use rv::misc::argmax;
use std::collections::VecDeque;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::marker::PhantomData;

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

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
    change_points
        .iter()
        .map(|cp| writeln!(cp_f, "{}", cp))
        .collect::<io::Result<()>>()?;

    Ok(())
}

/// Compute the locations in a sequence of distributions where the total
/// probability from 1 to window is above threshold.
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
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum ChangePointDetectionMethod {
    /// Detect when the most likely path is reset to zero.
    Reset,
    /// Detect when the most likely path is not incremented by one.
    NonIncremental,
    /// Detect when the most likely path's length drops by some fraction.
    DropThreshold(f64),
}

impl ChangePointDetectionMethod {
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
        }
        res
    }
}

/// Wrap a run-length detector to keep track of most likely breakpoints
#[derive(Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct MostLikelyPathWrapper<T, RL>
where
    RL: RunLengthDetector<T>,
{
    detector: RL,
    hist: VecDeque<f64>,
    _phantom_t: PhantomData<T>,
}

impl<T, RL> MapPathDetector<T> for MostLikelyPathWrapper<T, RL>
where
    RL: RunLengthDetector<T>,
{
    /// Step the underlying detector and return accumulated log probabilities
    ///
    /// Algorithm from: https://youtu.be/cas__TaFk9U
    fn step(&mut self, value: &T) -> MapPathResult {
        let run_length_probs = self.detector.step(value);
        let log_probs: Vec<f64> =
            run_length_probs.iter().map(|p| p.ln()).collect();

        let max_hist: f64 = self
            .hist
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            .clone();

        self.hist.push_front(max_hist);

        for (i, lp) in log_probs.iter().enumerate() {
            self.hist[i] += lp;
        }

        self.hist.truncate(log_probs.len());
        MapPathResult {
            map_path_probs: &self.hist,
            runlength_probs: run_length_probs,
        }
    }
}

impl<T, RL> MostLikelyPathWrapper<T, RL>
where
    RL: RunLengthDetector<T>,
{
    /// Create a new MostLikelyPathWrapper around a detector
    pub fn new(detector: RL) -> Self {
        Self {
            detector,
            hist: VecDeque::new(),
            _phantom_t: PhantomData,
        }
    }
}

/// Track the Maximum aposterori run-length for deviations
///
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct MapTracker {
    last_map: Option<usize>,
}

impl MapTracker {
    /// Create a new MapTracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Monitor a new run length or MAP path sequence of probabilities for changes
    pub fn track(&mut self, ps: &[f64]) -> MapChange {
        let this_map: usize = ps
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).expect("NaNs are not supported here")
            })
            .unwrap()
            .0;
        match self.last_map.replace(this_map) {
            None => MapChange::Initial(this_map),
            Some(last) => {
                if last + 1 == this_map {
                    MapChange::Incremental(this_map)
                } else {
                    MapChange::Interrupted(this_map)
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum MapChange {
    /// The first type of change from an initialization
    Initial(usize),
    /// MAP increased by one
    Incremental(usize),
    /// MAP changed by a value other than +1
    Interrupted(usize),
}

impl MapChange {
    pub fn location(&self) -> usize {
        match self {
            MapChange::Initial(s) => *s,
            MapChange::Incremental(s) => *s,
            MapChange::Interrupted(s) => *s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_change_simple() {
        let mut mc = MapTracker::new();
        assert_eq!(mc.track(&vec![1.0]), MapChange::Initial(0));
        assert_eq!(mc.track(&vec![0.25, 0.75]), MapChange::Incremental(1));
        assert_eq!(
            mc.track(&vec![0.25, 0.50, 0.25]),
            MapChange::Interrupted(1)
        );
    }
}
