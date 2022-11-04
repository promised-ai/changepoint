use changepoint::BocpdLike;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rv::dist::{
    Bernoulli, Beta, Gamma, Gaussian, MvGaussian, NormalGamma,
    NormalInvChiSquared, NormalInvGamma, NormalInvWishart, Poisson,
};
use rv::nalgebra::{DMatrix, DVector};

fn pyany_to_f64(x: &PyAny) -> PyResult<f64> {
    x.extract()
}

fn pyany_to_bool(x: &PyAny) -> PyResult<bool> {
    x.extract()
}

fn pyany_to_u32(x: &PyAny) -> PyResult<u32> {
    x.extract()
}

fn pyany_to_dvector(x: &PyAny) -> PyResult<DVector<f64>> {
    Python::with_gil(|py| {
        let np = PyModule::import(py, "numpy")?;
        let xs: Vec<f64> = np.getattr("array")?.call1((x,))?.extract()?;
        Ok(xs)
    })
    .map(DVector::from)
}

fn pyany_to_dmatrix(x: &PyAny) -> PyResult<DMatrix<f64>> {
    use numpy::PyArray2;
    Python::with_gil(|py| {
        let np = PyModule::import(py, "numpy")?;
        let xs: &PyArray2<f64> = np.getattr("array")?.call1((x,))?.extract()?;
        let shape = xs.shape();

        let data = unsafe {
            xs.as_slice().map_err(|_| {
                PyValueError::new_err("Non-contiguous memory error")
            })
        }?;

        let mat: DMatrix<f64> =
            DMatrix::from_row_slice(shape[0], shape[1], data);
        Ok(mat)
    })
}

#[derive(Clone, Debug)]
pub enum PriorVariant {
    NormalGamma(NormalGamma),
    NormalInvGamma(NormalInvGamma),
    NormalInvChiSquared(NormalInvChiSquared),
    NormalInvWishart(NormalInvWishart),
    Beta(Beta),
    Gamma(Gamma),
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Prior {
    pub dist: PriorVariant,
}

#[pymethods]
impl Prior {
    #[staticmethod]
    #[args(m = "0.0", r = "1.0", s = "1.0", v = "1.0")]
    pub fn normal_gamma(m: f64, r: f64, s: f64, v: f64) -> PyResult<Self> {
        NormalGamma::new(m, r, s, v)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::NormalGamma(dist),
            })
    }

    #[staticmethod]
    #[args(m = "0.0", v = "1.0", a = "1.0", b = "1.0")]
    pub fn normal_inv_gamma(m: f64, v: f64, a: f64, b: f64) -> PyResult<Self> {
        NormalInvGamma::new(m, v, a, b)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::NormalInvGamma(dist),
            })
    }

    #[staticmethod]
    #[args(m = "0.0", k = "1.0", v = "1.0", s2 = "1.0")]
    pub fn normal_inv_chi_squared(
        m: f64,
        k: f64,
        v: f64,
        s2: f64,
    ) -> PyResult<Self> {
        NormalInvChiSquared::new(m, k, v, s2)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::NormalInvChiSquared(dist),
            })
    }

    #[staticmethod]
    pub fn normal_inv_wishart(
        mu: &PyAny,
        k: f64,
        df: usize,
        scale: &PyAny,
    ) -> PyResult<Self> {
        let mu_vec = pyany_to_dvector(mu)?;
        let scale_mat = pyany_to_dmatrix(scale)?;
        NormalInvWishart::new(mu_vec, k, df, scale_mat)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::NormalInvWishart(dist),
            })
    }

    #[staticmethod]
    #[args(alpha = "0.5", beta = "0.5")]
    pub fn beta_bernoulli(alpha: f64, beta: f64) -> PyResult<Self> {
        Beta::new(alpha, beta)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::Beta(dist),
            })
    }

    #[staticmethod]
    #[args(shape = "1.0", rate = "1.0")]
    pub fn poisson_gamma(shape: f64, rate: f64) -> PyResult<Self> {
        Gamma::new(shape, rate)
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .map(|dist| Prior {
                dist: PriorVariant::Gamma(dist),
            })
    }
}

#[pyfunction]
#[pyo3(name = "NormalGamma")]
pub fn normal_gamma(m: f64, r: f64, s: f64, v: f64) -> PyResult<Prior> {
    Prior::normal_gamma(m, r, s, v)
}

#[pyfunction]
#[pyo3(name = "NormalInvGamma")]
pub fn normal_inv_gamma(m: f64, v: f64, a: f64, b: f64) -> PyResult<Prior> {
    Prior::normal_inv_gamma(m, v, a, b)
}

#[pyfunction]
#[pyo3(name = "NormalInvChiSquared")]
pub fn normal_inv_chi_squared(
    m: f64,
    k: f64,
    v: f64,
    s2: f64,
) -> PyResult<Prior> {
    Prior::normal_inv_chi_squared(m, k, v, s2)
}

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

#[pyfunction]
#[pyo3(name = "BetaBernoulli")]
pub fn beta_bernoulli(alpha: f64, beta: f64) -> PyResult<Prior> {
    Prior::beta_bernoulli(alpha, beta)
}

#[pyfunction]
#[pyo3(name = "PoissonGamma")]
pub fn poisson_gamma(shape: f64, rate: f64) -> PyResult<Prior> {
    Prior::poisson_gamma(shape, rate)
}

fn dist_to_bocpd(dist: &Prior, lambda: f64) -> BocpdVariant {
    match &dist.dist {
        PriorVariant::NormalGamma(pr) => BocpdVariant::NormalGamma(
            changepoint::Bocpd::new(lambda, pr.clone()),
        ),
        PriorVariant::NormalInvGamma(pr) => BocpdVariant::NormalInvGamma(
            changepoint::Bocpd::new(lambda, pr.clone()),
        ),
        PriorVariant::NormalInvChiSquared(pr) => {
            BocpdVariant::NormalInvChiSquared(changepoint::Bocpd::new(
                lambda,
                pr.clone(),
            ))
        }
        PriorVariant::NormalInvWishart(pr) => BocpdVariant::NormalInvWishart(
            changepoint::Bocpd::new(lambda, pr.clone()),
        ),
        PriorVariant::Beta(pr) => BocpdVariant::BetaBernoulli(
            changepoint::Bocpd::new(lambda, pr.clone()),
        ),
        PriorVariant::Gamma(pr) => BocpdVariant::PoissonGamma(
            changepoint::Bocpd::new(lambda, pr.clone()),
        ),
    }
}

#[derive(Clone, Debug)]
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

    fn step(&mut self, datum: &PyAny) -> PyResult<Vec<f64>> {
        match self {
            Self::NormalGamma(bocpd) => {
                let x = pyany_to_f64(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::NormalInvGamma(bocpd) => {
                let x = pyany_to_f64(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::NormalInvChiSquared(bocpd) => {
                let x = pyany_to_f64(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::BetaBernoulli(bocpd) => {
                let x = pyany_to_bool(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::PoissonGamma(bocpd) => {
                let x = pyany_to_u32(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
            Self::NormalInvWishart(bocpd) => {
                // FIXME: check cardinality
                let x = pyany_to_dvector(datum)?;
                Ok(bocpd.step(&x).to_vec())
            }
        }
    }
}

/// Online Bayesian Change Point Detection state container
#[pyclass]
#[derive(Clone, Debug)]
pub struct Bocpd {
    bocpd: BocpdVariant,
}

#[pymethods]
impl Bocpd {
    /// Create a new BOCPD
    ///
    /// Parameters
    /// ----------
    /// prior: Prior
    ///     The (conjugate) prior, which also describes the likelihood
    ///     distribution for the stream.
    /// lambda: float
    ///     Expected mean run length. A smaller value means changepoints are
    ///     believed to occur at shorter intervals.
    ///
    /// Raises
    /// ------
    /// ValueError: lambda <= 0.0
    #[new]
    pub fn new(prior: Prior, lambda: f64) -> PyResult<Self> {
        if lambda <= 0.0 {
            return Err(PyValueError::new_err(
                "lambda must be greater than zero",
            ));
        }

        Ok(Bocpd {
            bocpd: dist_to_bocpd(&prior, lambda),
        })
    }

    /// Reset the Bocpd to the new state
    pub fn reset(&mut self) {
        self.bocpd.reset()
    }

    /// Observe a new datum. Returns the runlength probabilities for each step.
    pub fn step(&mut self, datum: &PyAny) -> PyResult<Vec<f64>> {
        self.bocpd.step(datum).map(|rs| rs.to_vec())
    }
}
