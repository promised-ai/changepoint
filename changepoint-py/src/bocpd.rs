use changepoint::{Bocpd, BocpdLike};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rv::dist::{Gaussian, NormalGamma};

/// Online Bayesian Change Point Detection state container with Normal Gamma
/// prior
#[pyclass]
pub struct BocpdNg {
    bocpd: Bocpd<f64, Gaussian, NormalGamma>,
}

#[pymethods]
impl BocpdNg {
    /// Create a new BOCPD with Normal Gamma Prior
    ///
    /// Parameters
    /// ----------
    /// m: float
    ///     The prior mean
    /// r: float
    ///     Relative precision of μ versus data
    /// s: float
    ///     The mean of rho (the precision) is v/s.
    /// v: float
    ///     Degrees of freedom of precision of rho
    ///
    /// Raises
    /// ------
    /// ValueError: Invalid parameters
    #[new]
    #[args(m = "0.0", r = "1.0", s = "1.0", v = "1.0")]
    pub fn new(lambda: f64, m: f64, r: f64, s: f64, v: f64) -> PyResult<Self> {
        let pr = NormalGamma::new(m, r, s, v)
            .map_err(|err| PyValueError::new_err(format!("{}", err)))?;

        Ok(BocpdNg {
            bocpd: Bocpd::new(lambda, pr),
        })
    }

    pub fn reset(&mut self) {
        self.bocpd.reset()
    }

    pub fn step(&mut self, datum: f64) -> Vec<f64> {
        self.bocpd.step(&datum).to_vec()
    }
}
