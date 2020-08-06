/// Trait for run-length detectors
pub trait RunLengthDetector<T> {
    /// Update the run-length detector and return a sequence of run length probabilities.
    fn step(&mut self, value: &T) -> &[f64];
    /// Reset internal state, new run-lengths will refer to steps after this point.
    fn reset(&mut self);
    /// Get mean votes for change-points
    fn votes_mean(&self) -> &[f64];
    /// Get vote variance for change-points
    fn votes_var(&self) -> &[f64];
}
