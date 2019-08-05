//! This library provides Change Point Detection (CPD) tools such as
//!  * Online Bayesian CPD as `OBCPD`

mod online_bayesian;
pub use online_bayesian::*;

mod hazard;
pub use hazard::*;

pub mod generators;

pub mod utils;
