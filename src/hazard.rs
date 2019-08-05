//! Hazard functions.

/// A constant hazard function.
/// This is the hazard function that corresponds to a geometric distribution
/// with timescale λ.
pub fn constant_hazard(lambda: f64) -> impl Fn(usize) -> f64 {
    let inv_lambda = 1.0 / lambda;
    move |_: usize| inv_lambda
}
