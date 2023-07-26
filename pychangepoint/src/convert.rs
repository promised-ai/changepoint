use nalgebra::{DMatrix, DVector};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn pyany_to_f64(x: &PyAny) -> PyResult<f64> {
    x.extract()
}

pub(crate) fn pyany_to_bool(x: &PyAny) -> PyResult<bool> {
    x.is_true()
}

pub(crate) fn pyany_to_u32(x: &PyAny) -> PyResult<u32> {
    x.extract()
}

pub(crate) fn pyany_to_dvector(x: &PyAny) -> PyResult<DVector<f64>> {
    Python::with_gil(|py| {
        let np = PyModule::import(py, "numpy")?;
        let xs: Vec<f64> = np.getattr("array")?.call1((x,))?.extract()?;
        Ok(xs)
    })
    .map(DVector::from)
}

pub(crate) fn pyany_to_dmatrix(x: &PyAny) -> PyResult<DMatrix<f64>> {
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
