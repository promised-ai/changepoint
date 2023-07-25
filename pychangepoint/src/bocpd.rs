use crate::convert;
use changepoint::BocpdLike;
use nalgebra::DVector;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::PyTuple;
use rv::dist::{
    Bernoulli, Beta, Gamma, Gaussian, MvGaussian, NormalGamma,
    NormalInvChiSquared, NormalInvGamma, NormalInvWishart, Poisson,
};

use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};

/// The variant of the prior distribution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PriorVariant {
    NormalGamma(NormalGamma),
    NormalInvGamma(NormalInvGamma),
    NormalInvChiSquared(NormalInvChiSquared),
    NormalInvWishart(NormalInvWishart),
    Beta(Beta),
    Gamma(Gamma),
}

/// Prior distribution, which also describes the liklihood distribution of the
/// change point detector.
#[pyclass(module = "changepoint")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prior {
    pub dist: PriorVariant,
}

macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

macro_rules! handle_kind {
    ($name: ident, $args: ident, $($idx:tt $n:tt),+) => {{
        if $args.len() == count!($($idx)*) {
            Self::$name(
                $(
                    $args.get_item($idx)?.extract()?,
                )+
            )
        } else {
            Err(PyTypeError::new_err(format!("Prior kind '{}' requires the following arguments: {}", stringify!($name), stringify!($($n),+))))
        }
    }}
}

#[pymethods]
impl Prior {
    #[new]
    #[pyo3(signature = (kind, *args))]
    pub fn new(kind: &str, args: &PyTuple) -> PyResult<Self> {
        match kind {
            "normal_gamma" => {
                handle_kind!(normal_gamma, args, 0 m, 1 r, 2 s, 3 v)
            }
            "normal_inv_gamma" => {
                handle_kind!(normal_inv_gamma, args, 0 m, 1 v, 2 a, 3 b)
            }
            "normal_inv_chi_squared" => {
                handle_kind!(normal_inv_chi_squared, args, 0 m, 1 k, 2 v, 3 s2)
            }
            "normal_inv_wishart" => {
                handle_kind!(normal_inv_wishart, args, 0 m, 1 k, 2 df, 3 scale)
            }
            "beta_bernoulli" => {
                handle_kind!(beta_bernoulli, args, 0 alpha, 1 beta)
            }
            "poisson_gamma" => {
                handle_kind!(poisson_gamma, args, 0 shape, 1 rate)
            }
            unknown_kind => Err(PyTypeError::new_err(format!(
                "Unknown prior kind '{unknown_kind}'"
            ))),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (m = 0.0, r = 1.0, s = 1.0, v = 1.0))]
    pub fn normal_gamma(m: f64, r: f64, s: f64, v: f64) -> PyResult<Self> {
        NormalGamma::new(m, r, s, v)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::NormalGamma(dist),
            })
    }

    #[staticmethod]
    #[pyo3(signature = (m = 0.0, v = 1.0, a = 1.0, b = 1.0))]
    pub fn normal_inv_gamma(m: f64, v: f64, a: f64, b: f64) -> PyResult<Self> {
        NormalInvGamma::new(m, v, a, b)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))
            .map(|dist| Prior {
                dist: PriorVariant::NormalInvGamma(dist),
            })
    }

    #[staticmethod]
    #[pyo3(signature = (m = 0.0, k = 1.0, v = 1.0, s2 = 1.0))]
    pub fn normal_inv_chi_squared(
        m: f64,
        k: f64,
        v: f64,
        s2: f64,
    ) -> PyResult<Self> {
        NormalInvChiSquared::new(m, k, v, s2)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))
            .map(|dist| Prior {
                dist: PriorVariant::NormalInvChiSquared(dist),
            })
    }

    #[staticmethod]
    #[pyo3(signature = (mu, k, df, scale))]
    pub fn normal_inv_wishart(
        mu: &PyAny,
        k: f64,
        df: usize,
        scale: &PyAny,
    ) -> PyResult<Self> {
        let mu_vec = convert::pyany_to_dvector(mu)?;
        let scale_mat = convert::pyany_to_dmatrix(scale)?;
        NormalInvWishart::new(mu_vec, k, df, scale_mat)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::NormalInvWishart(dist),
            })
    }

    #[staticmethod]
    #[pyo3(signature = (alpha = 0.5, beta = 0.5))]
    pub fn beta_bernoulli(alpha: f64, beta: f64) -> PyResult<Self> {
        Beta::new(alpha, beta)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::Beta(dist),
            })
    }

    #[staticmethod]
    #[pyo3(signature = (shape = 1.0, rate = 1.0))]
    pub fn poisson_gamma(shape: f64, rate: f64) -> PyResult<Self> {
        Gamma::new(shape, rate)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::Gamma(dist),
            })
    }

    pub fn __repr__(&self) -> String {
        match &self.dist {
            PriorVariant::NormalGamma(ng) => {
                format!(
                    "Prior.NormalGamma(m = {}, s = {}, r = {}, v = {})",
                    ng.m(),
                    ng.s(),
                    ng.r(),
                    ng.v()
                )
            }
            PriorVariant::NormalInvGamma(nig) => {
                format!(
                    "Prior.NormalInvGamma(m = {}, v = {}, a = {}, b = {})",
                    nig.m(),
                    nig.v(),
                    nig.a(),
                    nig.b()
                )
            }
            PriorVariant::NormalInvChiSquared(nics) => {
                format!("Prior.NormalInvChiSquared(m = {}, k = {}, v = {}, s2 = {})", nics.m(), nics.k(), nics.v(), nics.s2())
            }
            PriorVariant::NormalInvWishart(niw) => format!(
                "Prior.NormalInvWishart(mu = {}, k = {}, df = {}, scale = {})",
                niw.mu(),
                niw.k(),
                niw.df(),
                niw.scale()
            ),
            PriorVariant::Beta(b) => format!(
                "Prior.Beta(alpha = {}, beta = {})",
                b.alpha(),
                b.beta()
            ),
            PriorVariant::Gamma(g) => format!(
                "Prior.Gamma(shape = {}, rate = {})",
                g.shape(),
                g.rate()
            ),
        }
    }

    pub fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.dist = deserialize(&state).unwrap();
        Ok(())
    }

    pub fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(serialize(&self.dist).unwrap())
    }

    pub fn __getnewargs__(&self) -> PyResult<(String, f64, f64, f64, f64)> {
        Ok(("normal_gamma".to_string(), 0.0, 1.0, 1.0, 1.0))
    }
}

/// Normal Gamma prior on univariate Normal random variable.
///
/// Parameters
/// ----------
/// m: float
///     Prior mean
/// r: float > 0.0
/// s: float > 0.0
/// v: float > 0.0
///
/// Returns
/// -------
/// Normal Gamma-Square prior
///
/// Raises
/// ------
/// ValueError:
///     - m, r, s, v is infinite or NaN,
///     - r, s, v <= 0.0
#[pyfunction(signature = (m = 0.0, r = 1.0, s = 1.0, v = 1.0))]
#[pyo3(
    name = "NormalGamma",
    text_signature = "(m = 0.0, r = 1.0, s = 1.0, v = 1.0)"
)]
pub fn normal_gamma(m: f64, r: f64, s: f64, v: f64) -> PyResult<Prior> {
    Prior::normal_gamma(m, r, s, v)
}

/// Normal Inverse-Gamma prior on univariate Normal random variable.
///
/// Parameters
/// ----------
/// m: float
///     Prior mean
/// v: float > 0.0
/// a: float > 0.0
/// b: float > 0.0
///
/// Returns
/// -------
/// Normal Inverse-Gamma prior
///
/// Raises
/// ------
/// ValueError:
///     - m, v, a, b is infinite or NaN,
///     - v, a, b <= 0.0
#[pyfunction(signature = (m = 0.0, v = 1.0, a = 1.0, b = 1.0))]
#[pyo3(
    name = "NormalInvGamma",
    text_signature = "(m = 0.0, v = 1.0, a = 1.0, b = 1.0)"
)]
pub fn normal_inv_gamma(m: f64, v: f64, a: f64, b: f64) -> PyResult<Prior> {
    Prior::normal_inv_gamma(m, v, a, b)
}

/// Normal Inverse-Chi-Squared prior on univariate Normal random variable.
///
/// Parameters
/// ----------
/// m: float
///     Prior mean
/// k: float > 0.0
///     Amount of confidence in the prior mean; measured in pseudo observations.
/// v: float > 0.0
///     Amount of confidence in the prior variance; measured in pseudo
///     observations.
/// s2: float > 0.0
///     Prior variance
///
/// Returns
/// -------
/// Normal Inverse-Chi-Square prior
///
/// Raises
/// ------
/// ValueError:
///     - m, k, v, s2 is infinite or NaN,
///     - k, v, s2 <= 0.0
#[pyfunction(signature = (m = 0.0, k = 1.0, v = 1.0, s2 = 1.0))]
#[pyo3(
    name = "NormalInvChiSquared",
    text_signature = "(m = 0.0, k = 1.0, v = 1.0, s2 = 1.0)"
)]
pub fn normal_inv_chi_squared(
    m: f64,
    k: f64,
    v: f64,
    s2: f64,
) -> PyResult<Prior> {
    Prior::normal_inv_chi_squared(m, k, v, s2)
}

/// Normal Inverse-Wishart prior on multivariate Normal random variable.
#[pyfunction]
#[pyo3(name = "NormalInvWishart")]
pub fn normal_inv_wishart(
    mu: &PyAny,
    k: f64,
    df: usize,
    scale: &PyAny,
) -> PyResult<Prior> {
    Prior::normal_inv_wishart(mu, k, df, scale)
}

/// Beta prior on a Bernoulli random variable.
///
/// Parameters
/// ----------
/// alpha: float > 0
///     alpha parameter of beta distribution
/// beta: float > 0
///     beta parameter of beta distribution
///
/// Returns
/// -------
/// Beta prior on Bernoulli distribution
///
/// Raises
/// ------
/// ValueError: alpha or beta is <= 0, infinite, or NaN
#[pyfunction(signature = (alpha = 0.5, beta = 0.5))]
#[pyo3(name = "BetaBernoulli", text_signature = "(alpha = 0.5, beta = 0.5)")]
pub fn beta_bernoulli(alpha: f64, beta: f64) -> PyResult<Prior> {
    Prior::beta_bernoulli(alpha, beta)
}

/// Gamma prior on a Poisson random variable.
///
/// Parameters
/// ----------
/// shape: float > 0
///     The Gamma distribution shape parameter
/// rate: float > 0
///     The Gamma distribution rate parameter
///
/// Returns
/// -------
/// Gamma prior on Poisson
///
/// Raises
/// ------
/// ValueError: shape or rate is <= 0, infinite, or NaN
#[pyfunction(signature = (shape = 1.0, rate = 1.0))]
#[pyo3(name = "PoissonGamma", text_signature = "(shape = 1.0, rate = 1.0)")]
pub fn poisson_gamma(shape: f64, rate: f64) -> PyResult<Prior> {
    Prior::poisson_gamma(shape, rate)
}

fn dist_to_bocpd(dist: Prior, lambda: f64) -> BocpdVariant {
    match dist.dist {
        PriorVariant::NormalGamma(pr) => {
            BocpdVariant::NormalGamma(changepoint::Bocpd::new(lambda, pr))
        }
        PriorVariant::NormalInvGamma(pr) => {
            BocpdVariant::NormalInvGamma(changepoint::Bocpd::new(lambda, pr))
        }
        PriorVariant::NormalInvChiSquared(pr) => {
            BocpdVariant::NormalInvChiSquared(changepoint::Bocpd::new(
                lambda, pr,
            ))
        }
        PriorVariant::NormalInvWishart(pr) => {
            BocpdVariant::NormalInvWishart(changepoint::Bocpd::new(lambda, pr))
        }
        PriorVariant::Beta(pr) => {
            BocpdVariant::BetaBernoulli(changepoint::Bocpd::new(lambda, pr))
        }
        PriorVariant::Gamma(pr) => {
            BocpdVariant::PoissonGamma(changepoint::Bocpd::new(lambda, pr))
        }
    }
}

/// The variant of the `Bocpd`. Describes the prior, likelihood, and the input
/// data type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BocpdVariant {
    NormalGamma(changepoint::Bocpd<f64, Gaussian, NormalGamma>),
    NormalInvGamma(changepoint::Bocpd<f64, Gaussian, NormalInvGamma>),
    NormalInvChiSquared(changepoint::Bocpd<f64, Gaussian, NormalInvChiSquared>),
    NormalInvWishart(
        changepoint::Bocpd<DVector<f64>, MvGaussian, NormalInvWishart>,
    ),
    BetaBernoulli(changepoint::Bocpd<bool, Bernoulli, Beta>),
    PoissonGamma(changepoint::Bocpd<u32, Poisson, Gamma>),
}

impl BocpdVariant {
    /// Reset the stream and learned parameters
    fn reset(&mut self) {
        match self {
            Self::NormalGamma(bocpd) => bocpd.reset(),
            Self::NormalInvGamma(bocpd) => bocpd.reset(),
            Self::NormalInvChiSquared(bocpd) => bocpd.reset(),
            Self::NormalInvWishart(bocpd) => bocpd.reset(),
            Self::BetaBernoulli(bocpd) => bocpd.reset(),
            Self::PoissonGamma(bocpd) => bocpd.reset(),
        }
    }

    /// Observe a new datum. Returns the run length probabilities for each step.
    fn step(&mut self, datum: &PyAny) -> PyResult<Vec<f64>> {
        match self {
            Self::NormalGamma(bocpd) => {
                let x = convert::pyany_to_f64(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::NormalInvGamma(bocpd) => {
                let x = convert::pyany_to_f64(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::NormalInvChiSquared(bocpd) => {
                let x = convert::pyany_to_f64(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::BetaBernoulli(bocpd) => {
                let x = convert::pyany_to_bool(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::PoissonGamma(bocpd) => {
                let x = convert::pyany_to_u32(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::NormalInvWishart(bocpd) => {
                // FIXME: check cardinality
                let x = convert::pyany_to_dvector(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
        }
    }
}

/// Online Bayesian Change Point Detection state container
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[pyclass(module = "changepoint")]
/// Create a new BOCPD
///
/// Parameters
/// ----------
/// prior: Prior
///     The (conjugate) prior, which also describes the likelihood
///     distribution for the stream.
/// lam: float
///     Expected mean run length. A smaller value means changepoints are
///     believed to occur at shorter intervals.
///
/// Raises
/// ------
/// ValueError: lam <= 0.0
///
pub struct Bocpd {
    bocpd: BocpdVariant,
}

#[pymethods]
impl Bocpd {
    #[new]
    /// Create a new BOCPD
    pub fn new(prior: Prior, lam: f64) -> PyResult<Self> {
        if lam <= 0.0 {
            return Err(PyValueError::new_err("lam must be greater than zero"));
        }

        Ok(Bocpd {
            bocpd: dist_to_bocpd(prior, lam),
        })
    }

    /// Reset the Bocpd to the new state
    pub fn reset(&mut self) {
        self.bocpd.reset()
    }

    /// Observe a new datum. Returns the run length probabilities for each step.
    pub fn step(&mut self, datum: &PyAny) -> PyResult<Vec<f64>> {
        self.bocpd.step(datum).map(|rs| rs.to_vec())
    }

    pub fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.bocpd = deserialize(&state).unwrap();
        Ok(())
    }

    pub fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(serialize(&self.bocpd).unwrap())
    }

    pub fn __getnewargs__(&self) -> PyResult<(Prior, f64)> {
        Ok((Prior::beta_bernoulli(0.5, 0.5).unwrap(), 1.0))
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Lt => Ok(false),
            CompareOp::Le => Ok(false),
            CompareOp::Eq => Ok(self.bocpd == other.bocpd),
            CompareOp::Ne => Ok(self.bocpd != other.bocpd),
            CompareOp::Gt => Ok(false),
            CompareOp::Ge => Ok(false),
        }
    }
}
