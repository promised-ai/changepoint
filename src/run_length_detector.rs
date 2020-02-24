/// Trait for run-length detectors
pub trait RunLengthDetector<T> {
    /// Update the run-length detector and return a sequence of run length probabilities.
    fn step(&mut self, value: &T) -> &[f64];
}
