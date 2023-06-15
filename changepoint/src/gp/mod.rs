//! Gaussian Process change point detection and related utilities

use std::{f64::consts::PI, ops::AddAssign};

use crate::BocpdLike;
use nalgebra::{
    allocator::Allocator, constraint::SameNumberOfRows,
    constraint::ShapeConstraint, storage::StorageMut, ComplexField, DMatrix,
    DVector, DefaultAllocator, Dim, Matrix, OMatrix, Scalar, Vector, U1,
};
use num_traits::Zero;
use rv::{
    prelude::{Gaussian, NormalGamma, Rv, StudentsT as RvStudentsT},
    process::gaussian::kernel::Kernel,
};
use special::Gamma;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Create a lag matrix of a particular order
fn ar_split<X>(obs: &[X], order: usize) -> DMatrix<X>
where
    X: Zero + std::fmt::Debug + std::cmp::PartialEq + Copy + 'static,
{
    DMatrix::from_fn(obs.len(), order, |i, j| {
        if i <= j {
            X::zero()
        } else {
            obs[i - j - 1]
        }
    })
}

fn truncate_r(x: &[f64], epsilon: f64) -> Vec<f64> {
    x.iter().rev().position(|p| p > &epsilon).map_or_else(
        || x.to_vec(),
        |last_larger| {
            let truncate_idx = x.len() - last_larger;
            let mut truncated = x.split_at(truncate_idx).0.to_vec();
            let z: f64 = truncated.iter().sum();
            truncated.iter_mut().for_each(|p| *p /= z);
            truncated
        },
    )
}

/// Logistic Hazard parameters with `compute`
///
/// LH(x, h, a, b) = logistic(h) * logistic(a * x + b)
///
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[must_use]
pub struct LogisticHazard {
    /// Logit scaled, scaling factor for the logistic hazard function (increasing increases hazard
    /// over the whole space).
    h: f64,
    /// Scale Term (Higher means the slope of the logistic is higher).
    a: f64,
    /// Translation term (increasing moves the logistic to the left).
    b: f64,
}

impl LogisticHazard {
    /// Create a new `LogisticHazard`
    pub fn new(h: f64, a: f64, b: f64) -> Self {
        Self { h, a, b }
    }

    /// Compute with the specified parameters
    #[must_use]
    pub fn compute(&self, i: f64) -> f64 {
        let h = logistic(self.h);
        let lp = logistic(self.a * i + self.b);
        h * lp
    }
}

#[inline]
fn logistic(x: f64) -> f64 {
    (1.0 + (-x).exp()).recip()
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
/// Autoregressive Gaussian Process Change Point detection
pub struct Argpcp<K>
where
    K: Kernel,
{
    /// Number of steps observed
    t: usize,
    /// Run length probabilities
    run_length_pr: Vec<f64>,
    /// Kernel Function
    kernel: K,
    /// Observations
    obs: Vec<f64>,
    /// Maximum autoregressive lag
    max_lag: usize,
    mrc: usize,
    u: DMatrix<f64>,
    /// Scale Gamma alpha parameter
    alpha0: f64,
    /// Scale Gamma beta parameter
    beta0: f64,
    last_nlml: DVector<f64>,
    /// Logistic Hazard function parameters
    log_hazard: LogisticHazard,
    preds: Vec<StudentT>,
    alpha: DMatrix<f64>,
    alpha_t: DMatrix<f64>,
    beta_t: DMatrix<f64>,
    epsilon: f64,
}

impl<K> Argpcp<K>
where
    K: Kernel,
{
    /// Create a new Argpcp
    ///
    /// # Arguments
    /// * `kernel` - Kernel Function.
    /// * `max_lag` - Maximum Autoregressive lag.
    /// * `alpha0` - Scale Gamma distribution alpha parameter.
    /// * `beta0` - Scale Gamma distribution beta parameter.
    /// * `h` - Hazard scale in logit units (more negative leads to larger run-lengths).
    /// * `a` - Roughtly the slope of the logistic hazard function (increasing increases variance
    /// of run-lengths).
    /// * `b` - The offset of the logistic hazard function.
    pub fn new(
        kernel: K,
        max_lag: usize,
        alpha0: f64,
        beta0: f64,
        h: f64,
        a: f64,
        b: f64,
    ) -> Self {
        Self {
            t: 0,
            run_length_pr: vec![1.0],
            kernel,
            obs: vec![],
            max_lag,
            mrc: 1,
            u: DMatrix::identity(1, 1),
            alpha0,
            beta0,
            last_nlml: DVector::zeros(1),
            log_hazard: LogisticHazard::new(h, a, b),
            preds: Vec::new(),
            alpha: DMatrix::zeros(0, 1),
            alpha_t: DMatrix::zeros(0, 1),
            beta_t: DMatrix::zeros(0, 1),
            epsilon: 1E-10,
        }
    }

    fn h(&self, x: f64) -> f64 {
        self.log_hazard.compute(x)
    }

    /// Predicitive Distributions
    pub fn predictive_dists(&self) -> &[StudentT] {
        &self.preds
    }

    /// Reset the internal state clearning all observed data
    pub fn clear(&mut self) {
        self.t = 0;
        self.run_length_pr = vec![1.0];
        self.obs = vec![];
        self.mrc = 1;
        self.u = DMatrix::identity(1, 1);
        self.last_nlml = DVector::zeros(1);
        self.preds = Vec::new();
        self.alpha = DMatrix::zeros(0, 1);
        self.alpha_t = DMatrix::zeros(0, 1);
        self.beta_t = DMatrix::zeros(0, 1);
    }
}

/// `StudentT` Rv
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[must_use]
pub struct StudentT {
    /// Inner studentsT
    pub st: RvStudentsT,
    /// Mean
    pub mean: f64,
    /// Sigma
    pub sigma: f64,
}

impl StudentT {
    /// Create a new `StudentT` distribution
    pub fn new(mean: f64, sigma2: f64, v: f64) -> Self {
        Self {
            st: RvStudentsT::new_unchecked(v),
            mean,
            sigma: sigma2.sqrt(),
        }
    }
}

impl Rv<f64> for StudentT {
    fn ln_f(&self, x: &f64) -> f64 {
        self.st.ln_f(&((x - self.mean) / self.sigma))
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> f64 {
        let s: f64 = self.st.draw(rng);
        s * self.sigma + self.mean
    }
}

impl<K> BocpdLike<f64> for Argpcp<K>
where
    K: Kernel,
{
    type Fx = Gaussian;
    type PosteriorPredictive = NormalGamma;

    #[allow(clippy::too_many_lines)]
    fn step(&mut self, value: &f64) -> &[f64] {
        self.obs.push(*value);
        self.t += 1;

        // Create a matrix (obs.len() + 1, max_lag)
        // Each row contains a timestep observation with the previous
        //   timestep observations
        // E.g.: observations [1, 2, 3,] with max lag 2 yields a matrix of
        // [[ 0  0 ]
        //  [ 1  0 ]
        //  [ 2  1 ]
        //  [ 3  2 ]]
        let x = ar_split(&self.obs, self.max_lag);

        // Reverse the order of this lag matrix up to the latest timestep
        let rev_x = {
            let cols = x.rows_range((self.t - self.mrc)..self.t - 1);
            DMatrix::from_fn(cols.nrows(), cols.ncols(), |i, j| {
                cols[(cols.nrows() - 1 - i, j)]
            })
        };

        // Calculate the prior variance of the latest test point
        let (test_cov, _) = self
            .kernel
            .covariance_with_gradient(&x.row(self.t - 1))
            .unwrap();
        let kss = test_cov[(0, 0)];

        // Calculate the cross-covariance between the latest test point
        // and the previous 'training' observations
        let kstar: DVector<f64> = self
            .kernel
            .covariance(&rev_x, &x.row(self.t - 1))
            .column(0)
            .into();

        // Calculate predictive distributions (Student's T)
        // Eq. 5.19 from Turner thesis
        self.preds = vec![StudentT::new(
            0.0,
            (self.beta0 / self.alpha0) * kss,
            2.0 * self.alpha0,
        )];
        if self.mrc > 1 {
            let vk = chol_solve(&self.u, &kstar);

            let vk_alpha_cs = col_cumsum(vk.component_mul(&self.alpha));
            let vk_vk_cs = col_cumsum(vk.component_mul(&vk));
            let beta_div_alpha = self.beta_t.component_div(&self.alpha_t);

            vk_alpha_cs
                .iter()
                .zip(beta_div_alpha.iter())
                .zip(vk_vk_cs.iter())
                .zip(self.alpha_t.iter())
                .for_each(|(((&vkac, &bda), &vvc), &at)| {
                    self.preds.push(StudentT::new(
                        vkac,
                        bda * (kss - vvc),
                        2.0 * at,
                    ));
                });
        }

        // Update the Gaussian process sub-evidence matrix (u)
        // Algorithm 4 from the Turner thesis
        let kss = kss.sqrt();
        let mut kstar = kstar / kss;
        let mut u = DMatrix::zeros(self.mrc, self.mrc);
        u[(0, 0)] = kss;
        if self.mrc > 1 {
            u.column_mut(0)
                .rows_range_mut(1..(self.mrc))
                .copy_from(&kstar.column(0));
            rank_one_update(&mut self.u, &mut kstar, -1.0);

            u.view_mut((1, 1), (self.u.nrows(), self.u.ncols()))
                .copy_from(&self.u);
        }
        self.u = u;

        // Update the alpha and beta parameters
        // Eq. 5.17 from Turner thesis
        let rev_y = DMatrix::from_iterator(
            self.mrc,
            1,
            self.obs
                .iter()
                .skip(self.t - self.mrc)
                .rev()
                .take(self.mrc)
                .copied(),
        );
        self.alpha = chol_solve(&self.u, &rev_y);
        let t: DMatrix<f64> = DMatrix::from_iterator(
            self.mrc,
            1,
            (1..=self.mrc).map(|x| x as f64),
        );
        self.alpha_t = t.scale(0.5).add_scalar(self.alpha0);
        self.beta_t = col_cumsum(self.alpha.map(|x| x * x))
            .scale(0.5)
            .add_scalar(self.beta0);

        // Calculate Negative Log Marginal Likelihood
        let ln_beta_stuff = self.beta_t.map(|bt| (bt / self.beta0).ln());
        let nlml_cur_a = self
            .alpha_t
            .component_mul(&ln_beta_stuff)
            .add_scalar(self.alpha0.ln_gamma().0);
        let nlml_cur_b = col_cumsum(self.u.diagonal().map(f64::ln))
            - self.alpha_t.map(|at| at.ln_gamma().0)
            + t.scale(0.5 * (2.0 * PI * self.beta0).ln());
        let nlml_cur = nlml_cur_a + nlml_cur_b;

        // Calculate the log probabilities
        let log_pred_probs = DMatrix::from_fn(nlml_cur.nrows(), 1, |i, _| {
            if i == 0 {
                -nlml_cur[0]
            } else {
                self.last_nlml[(i - 1, 0)] - nlml_cur[(i, 0)]
            }
        });

        self.last_nlml = nlml_cur;
        let pred_probs = log_pred_probs.map(f64::exp);

        // Calculate the Run Length Probabilities
        let mut next_r: Vec<f64> = (0..=self.mrc)
            .map(|i| {
                if i == 0 {
                    self.run_length_pr
                        .iter()
                        .enumerate()
                        .map(|(j, p)| {
                            p * pred_probs[j] * self.h((j + 1) as f64)
                        })
                        .sum()
                } else {
                    self.run_length_pr[i - 1]
                        * pred_probs[i - 1]
                        * (1.0 - self.h((i) as f64))
                }
            })
            .collect();

        // Normalize the run-length probabilities
        let z: f64 = next_r.iter().sum();
        for x in &mut next_r {
            *x /= z;
        }

        // Prune R according to a probability cutoff
        // NOTE: add a max length parameter?
        self.run_length_pr = truncate_r(&next_r, self.epsilon);
        self.mrc = self.run_length_pr.len();

        // Adjust other variables
        self.u = self.u.view((0, 0), (self.mrc - 1, self.mrc - 1)).into();
        self.last_nlml = self.last_nlml.rows_range(0..(self.mrc - 1)).into();
        self.alpha = self.alpha.view((0, 0), (self.mrc - 1, 1)).into();
        self.alpha_t = self.alpha_t.view((0, 0), (self.mrc - 1, 1)).into();
        self.beta_t = self.beta_t.view((0, 0), (self.mrc - 1, 1)).into();

        &self.run_length_pr
    }

    fn reset(&mut self) {
        todo!()
    }

    fn preload(&mut self, _data: &[f64]) {
        todo!()
    }

    fn pp(&self) -> Self::PosteriorPredictive {
        todo!()
    }
}

fn col_cumsum<N, R, C>(mat: OMatrix<N, R, C>) -> OMatrix<N, R, C>
where
    N: Scalar + AddAssign<N> + Zero + Copy,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    let r_out = R::from_usize(mat.nrows());
    let c_out = C::from_usize(mat.ncols());
    let it = (0..mat.ncols())
        .map::<Vec<N>, _>(move |col_idx| {
            let col = mat.column(col_idx);
            col.into_iter()
                .scan(N::zero(), |state, x| {
                    *state += *x;
                    Some(*state)
                })
                .collect()
        })
        .flat_map(std::iter::IntoIterator::into_iter);

    OMatrix::from_iterator_generic(r_out, c_out, it)
}

// From NAlgebra's Cholesky struct
fn rank_one_update<N, Dm, Sm, Rx, Sx>(
    chol: &mut Matrix<N, Dm, Dm, Sm>,
    x: &mut Vector<N, Rx, Sx>,
    sigma: N::RealField,
) where
    N: ComplexField + Copy,
    N::RealField: Copy,
    Dm: Dim,
    Rx: Dim,
    Sm: StorageMut<N, Dm, Dm>,
    Sx: StorageMut<N, Rx, U1>,
{
    // heavily inspired by Eigen's `llt_rank_update_lower` implementation https://eigen.tuxfamily.org/dox/LLT_8h_source.html
    let n = x.nrows();
    assert_eq!(
        n,
        chol.nrows(),
        "The input vector must be of the same size as the factorized matrix."
    );

    let mut beta = nalgebra::one::<N::RealField>();

    for j in 0..n {
        // updates the diagonal
        let diag = N::real(unsafe { *chol.get_unchecked((j, j)) });
        let diag2 = diag.powi(2);
        let xj = unsafe { *x.get_unchecked(j) };
        let sigma_xj2 = N::modulus_squared(xj) * sigma;
        let gamma = diag2 * beta + sigma_xj2;
        let new_diag = (diag2 + (sigma_xj2 / beta)).sqrt();
        unsafe { *chol.get_unchecked_mut((j, j)) = N::from_real(new_diag) };
        beta += sigma_xj2 / diag2;

        // updates the terms of L
        let mut xjplus = x.rows_range_mut(j + 1..);
        let mut col_j = chol.view_range_mut(j + 1.., j);
        xjplus.axpy(-xj / N::from_real(diag), &col_j, N::one());
        if gamma != nalgebra::zero::<N::RealField>() {
            col_j.axpy(
                N::from_real(new_diag * sigma / gamma) * N::conjugate(xj),
                &xjplus,
                N::from_real(new_diag / diag),
            );
        }
    }
}

/// From Nalgebra's Cholesky
fn chol_solve<N, Dm, Sm, R2, C2, S2>(
    chol: &Matrix<N, Dm, Dm, Sm>,
    b: &Matrix<N, R2, C2, S2>,
) -> OMatrix<N, R2, C2>
where
    N: ComplexField,
    Dm: Dim,
    Sm: StorageMut<N, Dm, Dm>,
    C2: Dim,
    R2: Dim,
    S2: StorageMut<N, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R2, Dm>,
    DefaultAllocator: Allocator<N, R2, C2>,
{
    let mut b = b.clone_owned();
    chol.solve_lower_triangular_mut(&mut b);
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generators, utils::map_changepoints};
    use rand::{prelude::SmallRng, SeedableRng};
    use rv::process::gaussian::kernel::{
        ConstantKernel, RBFKernel, WhiteKernel,
    };

    #[test]
    fn check_ar_split() {
        let input: &[i8] = &[1, 2, 3, 4, 5];
        let lagmat = ar_split(input, 3);
        let expected = DMatrix::from_row_slice(
            5,
            3,
            &[0, 0, 0, 1, 0, 0, 2, 1, 0, 3, 2, 1, 4, 3, 2],
        );
        assert_eq!(lagmat, expected);
    }

    fn logit(x: f64) -> f64 {
        -(x.recip() - 1.0).ln()
    }

    #[test]
    fn check_truncate() {
        let input: &[f64] = &[0.1, 0.1, 0.1, 0.5, 0.1];
        let truncated = truncate_r(input, 0.1);
        assert_eq!(truncated, vec![0.125, 0.125, 0.125, 0.625]);
    }

    #[test]
    fn detect_obvious_switch() {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 700,
        );

        let mut cpd = Argpcp::new(
            ConstantKernel::new_unchecked(1.0) * RBFKernel::new_unchecked(20.0)
                + WhiteKernel::new_unchecked(1.0),
            10,
            1.0,
            1.0,
            logit(1.0 / 500.0),
            1.0,
            1.0,
        );
        let mut rs: Vec<Vec<f64>> = vec![];

        for d in data {
            rs.push(cpd.step(&d).into());
        }

        let change_points = map_changepoints(&rs);

        assert_eq!(change_points.len(), 2);
        assert_eq!(change_points[0], 0);
        assert!((change_points[1] as i32 - 500_i32).abs() < 5);
    }
}
