//! This library provides Change Point Detection (CPD) tools such as
//!  * Online Bayesian CPD as `Bocpd`

mod bocpd;
pub use bocpd::*;

mod run_length_detector;
pub use run_length_detector::*;

mod bocpd_truncated;
pub use bocpd_truncated::*;

mod hazard;
pub use hazard::*;

pub mod generators;

pub mod utils;
