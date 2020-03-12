use std::collections::VecDeque;

/// Trait for run-length detectors
pub trait RunLengthDetector<T> {
    /// Update the run-length detector and return a sequence of run length probabilities.
    fn step(&mut self, value: &T) -> &[f64];
}

/// Response from MapPathDetectors
pub struct MapPathResult<'a> {
    /// Run Length probablilities
    pub runlength_probs: &'a [f64],
    /// Most likely path, based on history.
    pub map_path_probs: &'a VecDeque<f64>,
}

/// Trait for Maximum aposterori path detection
pub trait MapPathDetector<T> {
    fn step(&mut self, value: &T) -> MapPathResult;
}
