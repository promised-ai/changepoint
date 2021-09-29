//! This library provides Change Point Detection (CPD) tools such as
//!  * Online Bayesian CPD as `Bocpd`
//!  * Autoregressive Gaussian CPD as `Argpcpd`
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]

#[cfg(test)]
pub mod generators;

mod bocpd;
pub use bocpd::*;

mod traits;
pub use self::traits::*;

mod bocpd_truncated;
pub use bocpd_truncated::*;

pub mod gp;
pub use gp::*;

pub mod utils;

pub use rv;
