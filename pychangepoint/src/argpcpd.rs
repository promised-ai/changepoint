use bincode::{deserialize, serialize};
use changepoint::gp::Argpcp;
use changepoint::BocpdLike;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::PyBytes;
use rv::process::gaussian::kernel::{
    AddKernel, ConstantKernel, ProductKernel, RBFKernel, WhiteKernel,
};

/// Autoregressive Gaussian Process Change Point detection
///
/// Based on Ryan Turner's [thesis](https://www.repository.cam.ac.uk/bitstream/handle/1810/242181/thesis.pdf?sequence=1&isAllowed=y).
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
#[pyclass(module = "changepoint")]
pub struct ArgpCpd {
    argpcpd: Argpcp<
        AddKernel<ProductKernel<ConstantKernel, RBFKernel>, WhiteKernel>,
    >,
}

#[pymethods]
impl ArgpCpd {
    /// Create a new Argpcp
    #[new]
    #[pyo3(signature = (
        scale = 0.5,
        length_scale = 10.0,
        noise_level = 0.01,
        max_lag = 3,
        alpha0 = 2.0,
        beta0 = 1.0,
        logistic_hazard_h = -5.0,
        logistic_hazard_a = 1.0,
        logistic_hazard_b = 1.0
    ))]
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
    ///
    /// Parameters
    /// ----------
    ///
    /// datum: float
    ///     Next datum in the data stream.
    ///
    /// Returns
    /// -------
    /// List[float]
    ///     A list of change point probabilities.
    pub fn step(&mut self, datum: f64) -> Vec<f64> {
        self.argpcpd.step(&datum).to_vec()
    }

    pub fn __setstate__(
        &mut self,
        py: Python,
        state: PyObject,
    ) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.argpcpd = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.argpcpd).unwrap()).to_object(py))
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Lt => Ok(false),
            CompareOp::Le => Ok(false),
            CompareOp::Eq => Ok(self.argpcpd == other.argpcpd),
            CompareOp::Ne => Ok(self.argpcpd != other.argpcpd),
            CompareOp::Gt => Ok(false),
            CompareOp::Ge => Ok(false),
        }
    }
}
