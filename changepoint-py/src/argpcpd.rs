use changepoint::gp::Argpcp;
use changepoint::BocpdLike;
use pyo3::prelude::*;
use rv::process::gaussian::kernel::{
    AddKernel, ConstantKernel, ProductKernel, RBFKernel, WhiteKernel,
};

#[pyclass]
pub struct ArgpCpd {
    argpcpd: Argpcp<
        AddKernel<ProductKernel<ConstantKernel, RBFKernel>, WhiteKernel>,
    >,
}

#[pyclass]
#[derive(Clone)]
pub struct KernelArgs {
    #[pyo3(get, set)]
    scale: f64,
    #[pyo3(get, set)]
    length_scale: f64,
    #[pyo3(get, set)]
    noise_level: f64,
}

#[pymethods]
impl KernelArgs {
    #[new]
    pub fn new(scale: f64, length_scale: f64, noise_level: f64) -> Self {
        KernelArgs {
            scale,
            length_scale,
            noise_level,
        }
    }
}

#[pymethods]
impl ArgpCpd {
    #[new]
    pub fn new(
        kernel_args: KernelArgs,
        max_lag: usize,
        alpha0: f64,
        beta0: f64,
        logistic_hazard_h: f64,
        logistic_hazard_a: f64,
        logistic_hazard_b: f64,
    ) -> Self {
        let kernel = ConstantKernel::new_unchecked(kernel_args.scale)
            * RBFKernel::new_unchecked(kernel_args.length_scale)
            + WhiteKernel::new_unchecked(kernel_args.noise_level);

        ArgpCpd {
            argpcpd: Argpcp::new(
                kernel,
                max_lag,
                alpha0,
                beta0,
                logistic_hazard_h,
                logistic_hazard_a,
                logistic_hazard_b,
            ),
        }
    }

    pub fn reset(&mut self) {
        self.argpcpd.reset()
    }

    pub fn step(&mut self, datum: f64) -> Vec<f64> {
        self.argpcpd.step(&datum).to_vec()
    }
}
