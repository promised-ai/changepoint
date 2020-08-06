//! Online Bayesian Change Point Detection
//!
//! This code is derived from
//! "Bayesian Online Changepoint Detection"; Ryan Adams, David MacKay; arXiv:0710.3742
//! Which can be found [here](https://arxiv.org/pdf/0710.3742.pdf).

use crate::run_length_detector::RunLengthDetector;
use rv::prelude::*;
use std::collections::VecDeque;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Online Bayesian Change Point Detection state container
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Bocpd<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    hazard: f64,
    predictive_prior: Pr,
    suff_stats: VecDeque<Fx::Stat>,
    t: usize,
    r: Vec<f64>,
    empty_suffstat: Fx::Stat,
    cdf_threshold: f64,
    votes_mean: Vec<f64>,
    votes_var: Vec<f64>,
}

impl<X, Fx, Pr> Bocpd<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    /// Create a new Bocpd analyzer
    ///
    /// # Parameters
    /// * `hazard` - The hazard function for `P_{gap}`.
    /// * `fx` - Predictive distribution. Used for generating an empty `SuffStat`.
    /// * `predictive_prior` - Prior for the predictive distribution.
    ///
    /// # Example
    /// ```rust
    /// use changepoint::Bocpd;
    /// use rv::prelude::*;
    /// use std::sync::Arc;
    ///
    /// let cpd = Bocpd::new(
    ///     250.0,
    ///     Gaussian::standard(),
    ///     NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
    /// );
    /// ```
    pub fn new(hazard_lambda: f64, fx: Fx, predictive_prior: Pr) -> Self {
        Self {
            hazard: hazard_lambda.recip(),
            predictive_prior,
            suff_stats: VecDeque::new(),
            t: 0,
            r: Vec::new(),
            empty_suffstat: fx.empty_suffstat(),
            cdf_threshold: 1E-3,
            votes_var: Vec::new(),
            votes_mean: Vec::new(),
        }
    }

    /// Reset the introspector and replace the predictive prior
    pub fn reset_with_prior(&mut self, predictive_prior: Pr) {
        self.predictive_prior = predictive_prior;
        self.reset();
    }
}

impl<X, Fx, Pr> RunLengthDetector<X> for Bocpd<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    fn reset(&mut self) {
        self.suff_stats.clear();
        self.t = 0;
        self.r.clear();
    }

    /// Update the model with a new datum and return the distribution of run lengths.
    fn step(&mut self, data: &X) -> &[f64] {
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
                        .ln_pp(
                            data,
                            &DataOrSuffStat::SuffStat(&self.suff_stats[i]),
                        )
                        .exp();

                    r_seen += self.r[i];
                    let h = self.hazard;
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

        // Update the vote vecs
        self.votes_mean.push(0.0);
        self.votes_var.push(0.0);
        for i in 0..=self.t {
            let pi = self.r[self.t - i];
            self.votes_mean[i] += pi;
            self.votes_var[i] += (1.0 - pi) * pi;
        }

        // Update total sequence length
        self.t += 1;

        &self.r
    }

    fn votes_mean(&self) -> &[f64] {
        &self.votes_mean
    }
    fn votes_var(&self) -> &[f64] {
        &self.votes_var
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::ChangePointDetectionMethod;
    use crate::{generators, RunLengthDetector};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn each_vec_is_a_probability_dist() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 1000,
        );

        let mut cpd = Bocpd::new(
            250.0,
            Gaussian::standard(),
            NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(),
        );

        let res: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).to_vec()).collect();

        for row in res.iter() {
            let sum: f64 = row.iter().sum();
            assert::close(sum, 1.0, 1E-8);
        }
    }

    #[test]
    fn detect_obvious_switch() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 700,
        );

        let mut cpd = Bocpd::new(
            250.0,
            Gaussian::standard(),
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
        );

        let res: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).into()).collect();
        let change_points =
            ChangePointDetectionMethod::NegativeChangeInMAP.detect(&res);

        assert_eq!(change_points, vec![500]);
        assert::close(cpd.votes_mean()[499], 186.4, 1E-2);

        let mut votes: Vec<(usize, &f64)> =
            cpd.votes_mean().iter().enumerate().collect();
        votes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let maxes: Vec<usize> =
            votes.iter().rev().take(2).map(|x| x.0).collect();
        assert_eq!(maxes, [0, 499]);
    }

    #[test]
    fn coal_mining_data() {
        let data = generators::coal_mining_incidents();

        let mut cpd = Bocpd::new(
            100.0,
            Poisson::new_unchecked(123.0),
            Gamma::new_unchecked(1.0, 1.0),
        );

        let res: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).into()).collect();
        let change_points =
            ChangePointDetectionMethod::NegativeChangeInMAP.detect(&res);

        assert_eq!(change_points, vec![47, 49, 53, 102]);
    }

    /// This test checks for change points with 3-month treasury bill market data
    ///
    /// # Data Source
    /// > Board of Governors of the Federal Reserve System (US), 3-Month Treasury Bill: Secondary
    /// > Market Rate [TB3MS], retrieved from FRED, Federal Reserve Bank of St. Louis;
    /// > https://fred.stlouisfed.org/series/TB3MS, March 24, 2020.
    #[test]
    fn treasury_changes() {
        let raw_data: &str = include_str!("../resources/TB3MS.csv");
        let data: Vec<f64> = raw_data
            .lines()
            .skip(1)
            .map(|line| {
                let (_, line) = line.split_at(11);
                line.parse().unwrap()
            })
            .collect();

        let mut cpd = Bocpd::new(
            250.0,
            Gaussian::standard(),
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1E-5),
        );

        let res: Vec<Vec<f64>> = data
            .iter()
            .zip(data.iter().skip(1))
            .map(|(a, b)| (b - a) / a)
            .map(|d| cpd.step(&d).into())
            .collect();
        let change_points =
            ChangePointDetectionMethod::NegativeChangeInMAP.detect(&res);

        assert_eq!(change_points, vec![93, 130, 294, 896, 981, 1005]);
    }
}
