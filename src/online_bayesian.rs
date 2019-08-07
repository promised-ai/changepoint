//! Online Bayesian Change Point Detection
//!
//! This code is derived from
//! "Bayesian Online Changepoint Detection"; Ryan Adams, David MacKay; arXiv:0710.3742
//! Which can be found [here](https://arxiv.org/pdf/0710.3742.pdf).

use rv::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
// use crate::sparse::SparseVec;

/// Online Bayesian Change Point Detection state container
pub struct BOCPD<X, H, Fx, Pr>
where
    H: Fn(usize) -> f64,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    hazard: H,
    predictive_prior: Arc<Pr>,
    suff_stats: VecDeque<Fx::Stat>,
    t: usize,
    r: Vec<f64>,
    empty_suffstat: Fx::Stat,
    cdf_threshold: f64,
}

impl<X, H, Fx, Pr> BOCPD<X, H, Fx, Pr>
where
    H: Fn(usize) -> f64,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    /// Create a new BOCPD analyzer
    ///
    /// # Parameters
    /// * `hazard` - The hazard function for `P_{gap}`.
    /// * `fx` - Predictive distribution. Used for generating an empty `SuffStat`.
    /// * `predictive_prior` - Prior for the predictive distribution.
    ///
    /// # Example
    /// ```rust
    /// use changepoint::{BOCPD, constant_hazard};
    /// use rv::prelude::*;
    /// use std::sync::Arc;
    ///
    /// let cpd = BOCPD::new(
    ///     constant_hazard(250.0),
    ///     &Gaussian::standard(),
    ///     Arc::new(NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap()),
    /// );
    /// ```
    pub fn new(hazard: H, fx: &Fx, predictive_prior: Arc<Pr>) -> Self {
        Self {
            hazard,
            predictive_prior,
            suff_stats: VecDeque::new(),
            t: 0,
            r: Vec::new(),
            empty_suffstat: fx.empty_suffstat(),
            cdf_threshold: 1E-3,
        }
    }

    /// Update the model with a new datum and return the distribution of run lengths.
    pub fn step(&mut self, data: &X) -> Vec<f64> {
        self.suff_stats.push_front(self.empty_suffstat.clone());

        if self.t == 0 {
            // The initial point is, by definition, a change point
            self.r.push(1.0);
        } else {
            self.r.push(0.0);
            let mut r0 = 0.0;
            let mut r_sum = 0.0;
            let mut r_seen = 0.0;

            for i in (0..self.t).rev() {
                if self.r[i] == 0.0 {
                    self.r[i + 1] = 0.0;
                } else {
                    // Evaluate growth probabilites and shift probabilities down
                    // scaling by the hazard function and the predprobs
                    let pp = self
                        .predictive_prior
                        .ln_pp(data, &DataOrSuffStat::SuffStat(&self.suff_stats[i]))
                        .exp();

                    r_seen += self.r[i];
                    let h = (self.hazard)(i);
                    self.r[i + 1] = self.r[i] * pp * (1.0 - h);
                    r0 += self.r[i] * pp * h;
                    r_sum += self.r[i + 1];

                    if 1.0 - r_seen < self.cdf_threshold {
                        break;
                    }
                }
            }
            r_sum += r0;
            // Accumulate mass back down to r[0], the probability there was a
            // change point at this location.
            self.r[0] = r0;

            // Normalize R
            for i in 0..=self.t {
                self.r[i] /= r_sum;
            }
        }

        // Update the SuffStat with the new data
        self.suff_stats
            .iter_mut()
            .for_each(|stat| stat.observe(data));

        // Update total sequence length
        self.t = self.t + 1;

        self.r.clone().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constant_hazard;
    use crate::generators;
    use crate::utils;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn each_vec_is_a_probability_dist() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(&mut rng, 0.0, 1.0, 10.0, 5.0, 500, 1000);

        let mut cpd = BOCPD::new(
            constant_hazard(250.0),
            &Gaussian::standard(),
            Arc::new(NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap()),
        );

        let res: Vec<Vec<f64>> = data.iter().map(|d| cpd.step(d)).collect();

        for row in res.iter() {
            let sum: f64 = row.iter().sum();
            assert::close(sum, 1.0, 1E-8);
        }
    }

    #[test]
    fn detect_obvious_switch() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(&mut rng, 0.0, 1.0, 10.0, 5.0, 500, 1000);

        let mut cpd = BOCPD::new(
            constant_hazard(250.0),
            &Gaussian::standard(),
            Arc::new(NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap()),
        );

        let res: Vec<Vec<f64>> = data.iter().map(|d| cpd.step(d)).collect();

        let change_points =
            utils::most_likely_breaks(&res, utils::ChangePointDetectionMethod::NonIncremental);

        // Write output
        // utils::write_data_and_r("obvious_jump", &data, &res, &change_points).unwrap();

        assert_eq!(change_points, vec![500]);
    }

    #[test]
    fn coal_mining_data() {
        let data = generators::coal_mining_incidents();

        let mut cpd: BOCPD<u8, _, Poisson, Gamma> = BOCPD::new(
            constant_hazard(100.0),
            &Poisson::new(123.0).unwrap(),
            Arc::new(Gamma::new(1.0, 1.0).unwrap()),
        );

        let res: Vec<Vec<f64>> = data.iter().map(|d| cpd.step(d)).collect();

        let change_points =
            utils::most_likely_breaks(&res, utils::ChangePointDetectionMethod::DropThreshold(0.5));

        // Write output
        // utils::write_data_and_r("mining", &data, &res, &change_points).unwrap();

        assert_eq!(change_points, vec![50, 107]);
    }

    /// This test checks the BOCPD algorithm against S&P/Case-Schiller 20-City
    /// Composite Home Price Index.
    ///
    /// # Data Source
    /// > S&P Dow Jones Indices LLC, S&P/Case-Shiller 20-City Composite Home Price Index
    /// > [SPCS20RSA], retrieved from FRED, Federal Reserve Bank of St. Louis;
    /// > https://fred.stlouisfed.org/series/SPCS20RSA, August 5, 2019.
    #[test]
    fn housing_change() {
        let raw_data: &str = include_str!("../resources/SPCS20RSA.csv");
        let data: Vec<f64> = raw_data
            .lines()
            .skip(1)
            .map(|line| {
                let (_, line) = line.split_at(11);
                line.parse().unwrap()
            })
            .collect();

        let mut cpd = BOCPD::new(
            constant_hazard(250.0),
            &Gaussian::standard(),
            Arc::new(NormalGamma::new(0.0, 1.0, 1.0, 1E-5).unwrap()),
        );

        let res: Vec<Vec<f64>> = data.iter().map(|d| cpd.step(d)).collect();

        let change_points =
            utils::most_likely_breaks(&res, utils::ChangePointDetectionMethod::DropThreshold(0.1));

        // Write output
        // utils::write_data_and_r("housing_change", &data, &res, &change_points).unwrap();

        assert_eq!(change_points, vec![81, 156]);
    }
}
