pub mod argpcpd;
pub mod bocpd;

use changepoint::utils;
use pyo3::prelude::*;

#[pymodule]
fn pychangepoint(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<bocpd::BocpdNg>()?;
    m.add_class::<argpcpd::ArgpCpd>()?;

    #[pyfn(m)]
    #[pyo3(name = "infer_changepoints")]
    fn infer_changepoints(rs: Vec<Vec<f64>>, sample_size: u32) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        utils::infer_changepoints(&rs, sample_size as usize, &mut rng).unwrap()
    }

    #[pyfn(m)]
    #[pyo3(name = "infer_pseudo_cmf_changepoints")]
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

    #[pyfn(m)]
    #[pyo3(name = "map_changepoints")]
    fn map_changepoints(rs: Vec<Vec<f64>>) -> Vec<usize> {
        utils::map_changepoints(&rs)
    }

    Ok(())
}
