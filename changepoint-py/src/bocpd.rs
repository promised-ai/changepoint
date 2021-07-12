use changepoint::{Bocpd, BocpdLike};
use pyo3::prelude::*;
use rv::dist::{Gaussian, NormalGamma};

#[pyclass]
pub struct BocpdNg {
    bocpd: Bocpd<f64, Gaussian, NormalGamma>,
}

#[pyclass]
#[derive(Clone)]
pub struct Ng {
    #[pyo3(get, set)]
    m: f64,
    #[pyo3(get, set)]
    r: f64,
    #[pyo3(get, set)]
    s: f64,
    #[pyo3(get, set)]
    v: f64,
}

#[pymethods]
impl Ng {
    #[new]
    pub fn new(m: f64, r: f64, s: f64, v: f64) -> Self {
        Ng { m, r, s, v }
    }
}

#[pymethods]
impl BocpdNg {
    #[new]
    pub fn new(lambda: f64, ng: Ng) -> Self {
        let pr = NormalGamma::new_unchecked(ng.m, ng.r, ng.s, ng.v);
        BocpdNg {
            bocpd: Bocpd::new(lambda, pr),
        }
    }

    pub fn reset(&mut self) {
        self.bocpd.reset()
    }

    pub fn step(&mut self, datum: f64) -> Vec<f64> {
        self.bocpd.step(&datum).to_vec()
    }
}
