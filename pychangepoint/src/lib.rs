pub mod argpcpd;
pub mod bocpd;
pub(crate) mod convert;

use changepoint::utils;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "changepoint")]
fn core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<bocpd::Prior>()?;
    m.add_class::<bocpd::Bocpd>()?;
    m.add_class::<argpcpd::ArgpCpd>()?;
    m.add_function(wrap_pyfunction!(bocpd::normal_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bocpd::normal_inv_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bocpd::normal_inv_chi_squared, m)?)?;
    m.add_function(wrap_pyfunction!(bocpd::normal_inv_wishart, m)?)?;
    m.add_function(wrap_pyfunction!(bocpd::poisson_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bocpd::beta_bernoulli, m)?)?;

    /// Given the runlength probabilities vector for each step, and a Monte
    /// Carlo step size, return the probability that a change point occurred on
    /// a given step.
    ///
    /// Parameters
    /// ----------
    /// history: List[List[float]]
    ///     Run length probabilities for a sequence of observations.
    /// n_samples: int
    ///     Number of samples to draw from run-length process.
    #[pyfn(m)]
    #[pyo3(
        name = "infer_changepoints",
        text_signature = "(history, n_samples)"
    )]
    fn infer_changepoints(rs: Vec<Vec<f64>>, sample_size: u32) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        utils::infer_changepoints(&rs, sample_size as usize, &mut rng).unwrap()
    }

    /// Creates a pseudo cmf distribution for change-point locations.
    ///
    /// This calculates the cumulative sum of the `infer_changepoints` return
    /// value mod 1.0.
    ///
    /// Parameters
    /// ----------
    /// history: List[List[float]]
    ///     Run length probabilities for a sequence of observations.
    /// n_samples: int
    ///     Number of samples to draw from run-length process.
    #[pyfn(m)]
    #[pyo3(
        name = "infer_pseudo_cmf_changepoints",
        text_signature = "(history, n_samples)"
    )]
    fn infer_pseudo_cmf_changepoints(
        rs: Vec<Vec<f64>>,
        sample_size: u32,
    ) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        utils::infer_pseudo_cmf_changepoints(
            &rs,
            sample_size as usize,
            &mut rng,
        )
        .unwrap()
    }

    /// Returns the most likely index (step) that a changepoint occurred. Each
    /// point corresponds to an individual changepoint.
    ///
    /// Parameters
    /// ----------
    /// history: List[List[float]]
    ///     Run length probabilities for a sequence of observations.
    #[pyfn(m)]
    #[pyo3(name = "map_changepoints", text_signature = "(history)")]
    fn map_changepoints(rs: Vec<Vec<f64>>) -> Vec<usize> {
        utils::map_changepoints(&rs)
    }

    Ok(())
}
