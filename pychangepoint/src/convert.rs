use nalgebra::{DMatrix, DVector};
use numpy::{PyArrayLike1, PyArrayLike2, PyUntypedArrayMethods, TypeMustMatch};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn pyany_to_f64<'py>(x: &'py Bound<'py, PyAny>) -> PyResult<f64> {
    x.extract()
}

pub(crate) fn pyany_to_bool<'py>(x: &'py Bound<'py, PyAny>) -> PyResult<bool> {
    x.is_truthy()
}

pub(crate) fn pyany_to_u32<'py>(x: &'py Bound<'py, PyAny>) -> PyResult<u32> {
    x.extract()
}

pub(crate) fn pyarray1_to_dvector<'py>(
    x: PyArrayLike1<'py, f64, TypeMustMatch>,
) -> PyResult<DVector<f64>> {
    let data = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("Non-contiguous memory error"))?;

    let mat: DVector<f64> = DVector::from_column_slice(data);
    Ok(mat)
}

pub(crate) fn pyarray2_to_dmatrix<'py>(
    array: PyArrayLike2<'py, f64, TypeMustMatch>,
) -> PyResult<DMatrix<f64>> {
    let shape = array.shape();
    let data = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("Non-contiguous memory error"))?;

    let mat: DMatrix<f64> = DMatrix::from_row_slice(shape[0], shape[1], data);
    Ok(mat)
}
