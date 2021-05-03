//! This library provides Change Point Detection (CPD) tools such as
//!  * Online Bayesian CPD as `Bocpd`
//!  * Autoregressive Gaussian CPD as `Argpcpd`
#![warn(missing_docs)]

// Test the README
use doc_comment::doctest;
doctest!("../README.md");

mod bocpd;
pub use bocpd::*;

mod traits;
pub use self::traits::*;

mod bocpd_truncated;
pub use bocpd_truncated::*;

pub mod gp;
pub use gp::*;

#[cfg(test)]
pub mod generators;

pub mod utils;

pub use rv;
