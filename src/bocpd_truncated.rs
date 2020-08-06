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

/// Online Bayesian Change Point Detection with a truncated tail
///
/// The truncation takes place after run length probabilites are computed.
/// The truncation point is chosen based on the most recent point from which
/// all successive mass is below the given threshold.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct BocpdTruncated<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    hazard: f64,
    predictive_prior: Pr,
    suff_stats: VecDeque<Fx::Stat>,
    r: Vec<f64>,
    empty_suffstat: Fx::Stat,
    cdf_threshold: f64,
    cutoff_threadhold: f64,
    votes_mean: Vec<f64>,
    votes_var: Vec<f64>,
}

impl<X, Fx, Pr> BocpdTruncated<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    /// Create a new Bocpd analyzer
    ///
    /// # Parameters
    /// * `hazard` - The hazard function for `P_{gap} = 1/hazard`.
    /// * `fx` - Predictive distribution. Used for generating an empty `SuffStat`.
    /// * `predictive_prior` - Prior for the predictive distribution.
    ///
    /// # Example
    /// ```rust
    /// use changepoint::BocpdTruncated;
    /// use rv::prelude::*;
    /// use std::sync::Arc;
    ///
    /// let cpd = BocpdTruncated::new(
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
            r: Vec::new(),
            empty_suffstat: fx.empty_suffstat(),
            cdf_threshold: 1E-3,
            cutoff_threadhold: 1E-6,
            votes_mean: Vec::new(),
            votes_var: Vec::new(),
        }
    }

    /// Change the cutoff for mass to be discarded on the tail end of run-lengths
    pub fn with_cutoff(self, cutoff_threadhold: f64) -> Self {
        Self {
            cutoff_threadhold,
            ..self
        }
    }

    /// Reset the introspector and replace the predictive prior
    pub fn reset_with_prior(&mut self, predictive_prior: Pr) {
        self.predictive_prior = predictive_prior;
        self.reset();
    }
}

impl<X, Fx, Pr> RunLengthDetector<X> for BocpdTruncated<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    /// Update the model with a new datum and return the distribution of run lengths.
    fn step(&mut self, data: &X) -> &[f64] {
        self.suff_stats.push_front(self.empty_suffstat.clone());

        if self.r.is_empty() {
            // The initial point is, by definition, a change point
            self.r.push(1.0);
        } else {
            self.r.push(0.0);
            let mut r0 = 0.0;
            let mut r_sum = 0.0;
            let mut r_seen = 0.0;

            for i in (0..(self.r.len() - 1)).rev() {
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
            for i in 0..self.r.len() {
                self.r[i] /= r_sum;
            }

            // Truncate
            let cutoff = self
                .r
                .iter()
                .rev()
                .scan(0.0, |acc, p| {
                    *acc += p;
                    Some(*acc)
                })
                .enumerate()
                .find(|(_, cdf)| *cdf > self.cutoff_threadhold)
                .map(|x| self.r.len() - x.0 + 1);

            if let Some(trunc_index) = cutoff {
                self.r.truncate(trunc_index);
                self.suff_stats.truncate(trunc_index);
            }
        }

        // Update the SuffStat with the new data
        self.suff_stats
            .iter_mut()
            .for_each(|stat| stat.observe(data));

        // Update the vote vecs
        self.votes_mean.push(0.0);
        self.votes_var.push(0.0);

        let r_offset = self.votes_mean.len() - self.r.len();
        let r_len = self.r.len();
        for i in 0..r_len {
            let pi = self.r[r_len - i - 1];
            self.votes_mean[i + r_offset] += pi;
            self.votes_var[i + r_offset] += (1.0 - pi) * pi;
        }

        debug_assert!(
            !self.r.iter().any(|x| x.is_nan()),
            "Resulting run-length probabilities cannot contain NaNs"
        );
        &self.r
    }

    fn reset(&mut self) {
        self.suff_stats.clear();
        self.r.clear();
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
    use crate::generators;
    use crate::utils::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn each_vec_is_a_probability_dist() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 1000,
        );

        let mut cpd = BocpdTruncated::new(
            250.0,
            Gaussian::standard(),
            NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(),
        );

        let res: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).to_vec()).collect();

        for row in res.iter() {
            let sum: f64 = row.iter().sum();
            assert::close(sum, 1.0, 1E-6);
        }
    }

    #[test]
    fn detect_obvious_switch() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 700,
        );

        let mut cpd = BocpdTruncated::new(
            250.0,
            Gaussian::standard(),
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
        );

        let res: Vec<Vec<f64>> =
            data.iter().map(|d| (*cpd.step(d)).into()).collect();
        let change_points =
            ChangePointDetectionMethod::NegativeChangeInMAP.detect(&res);

        assert_eq!(change_points, vec![500]);

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

        let mut cpd = BocpdTruncated::new(
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

        let mut cpd = BocpdTruncated::new(
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
            ChangePointDetectionMethod::DropThreshold(0.1).detect(&res);

        assert_eq!(change_points, vec![93, 130, 294, 896, 981, 1005]);
    }
}
