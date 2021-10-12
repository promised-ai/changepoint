use changepoint::gp::Argpcp;
use changepoint::BocpdLike;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rv::process::gaussian::kernel::{
    AddKernel, ConstantKernel, ProductKernel, RBFKernel, WhiteKernel,
};

/// Autoregressive Gaussian Process Change Point detection
///
/// Based on Ryan Turner's [thesis](https://www.repository.cam.ac.uk/bitstream/handle/1810/242181/thesis.pdf?sequence=1&isAllowed=y).
#[pyclass]
pub struct ArgpCpd {
    argpcpd: Argpcp<
        AddKernel<ProductKernel<ConstantKernel, RBFKernel>, WhiteKernel>,
    >,
}

#[pymethods]
impl ArgpCpd {
    /// Create a new Argpcp
    ///
    /// Parameters
    /// ----------
    /// scale: float
    ///     Scale of the `ConstantKernel`
    /// length_scale:float
    ///     Length Scale of `RBFKernel`
    /// noise_level: float
    ///     Noise standard deviation for the `WhiteKernel`
    /// max_lag: int > 0
    ///     Maximum Autoregressive lag
    /// alpha0 : float
    ///     Scale Gamma distribution alpha parameter
    /// beta0: float
    ///     Scale Gamma distribution beta parameter
    /// logistic_hazard_h: float
    ///     Hazard scale in logit units.
    /// logistic_hazard_a: float
    ///     Roughly the slope of the logistic hazard function
    /// logistic_hazard_b: float
    ///     The offset of the logistic hazard function.
    #[new]
    #[args(
        scale = "0.5",
        length_scale = "10.0",
        noise_level = "0.01",
        max_lag = "3",
        alpha0 = "2.0",
        beta0 = "1.0",
        logistic_hazard_h = "-5.0",
        logistic_hazard_a = "1.0",
        logistic_hazard_b = "1.0"
    )]
    pub fn new(
        scale: f64,
        length_scale: f64,
        noise_level: f64,
        max_lag: usize,
        alpha0: f64,
        beta0: f64,
        logistic_hazard_h: f64,
        logistic_hazard_a: f64,
        logistic_hazard_b: f64,
    ) -> PyResult<Self> {
        let constant_kernel = ConstantKernel::new(scale)
            .map_err(|err| PyValueError::new_err(format!("{}", err)))?;
        let rbf_kernel = RBFKernel::new(length_scale)
            .map_err(|err| PyValueError::new_err(format!("{}", err)))?;
        let white_kernel = WhiteKernel::new(noise_level)
            .map_err(|err| PyValueError::new_err(format!("{}", err)))?;

        let kernel = constant_kernel * rbf_kernel + white_kernel;

        Ok(ArgpCpd {
            argpcpd: Argpcp::new(
                kernel,
                max_lag,
                alpha0,
                beta0,
                logistic_hazard_h,
                logistic_hazard_a,
                logistic_hazard_b,
            ),
        })
    }

    /// Reset the argpcpd to starting state
    pub fn reset(&mut self) {
        self.argpcpd.reset()
    }

    /// Observe a new datum and return the run length probabilities
    pub fn step(&mut self, datum: f64) -> Vec<f64> {
        self.argpcpd.step(&datum).to_vec()
    }
}
