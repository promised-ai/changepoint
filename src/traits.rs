use rv::prelude::Rv;

/// Trait for implementors of Bayesian online change-point detection
pub trait BocpdLike<T> {
    /// Type of type of distribution
    type Fx: Rv<T>;
    /// Type of predictive prior distribution
    type PosteriorPredictive: Rv<Self::Fx>;
    /// Update the run-length detector and return a sequence of run length probabilities.
    fn step(&mut self, value: &T) -> &[f64];
    /// Reset internal state, new run-lengths will refer to steps after this point.
    fn reset(&mut self);
    /// Generate the posterior predictive distribution
    fn pp(&self) -> Self::PosteriorPredictive;
    /// Preload a seqeunce into the default suff stat
    fn preload(&mut self, data: &[T]);
}
