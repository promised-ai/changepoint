//! Online Bayesian Change Point Detection
//!
//! This code is derived from
//! "Bayesian Online Changepoint Detection"; Ryan Adams, David `MacKay`; arXiv:0710.3742
//! Which can be found [here](https://arxiv.org/pdf/0710.3742.pdf).

use crate::traits::BocpdLike;
use rand::{rngs::SmallRng, SeedableRng};
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
#[derive(Clone, Debug)]
pub struct BocpdTruncated<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    /// Geometric distribution timescale used as the Hazard Function.
    /// \[
    ///     H(\tao) = 1/\lambda
    /// \] where `hazard` is $\lambda$.
    hazard: f64,
    /// Probability of observing data type `X` i.e. $\pi^{(r)}_t = P(x_t | suff_stat)$.
    predictive_prior: Pr,
    /// Sequence of sufficient statistics representing cumulative states.
    suff_stats: VecDeque<Fx::Stat>,
    /// Initial suff_stat
    initial_suffstat: Option<Fx::Stat>,
    /// Run-length probabilities.
    r: Vec<f64>,
    empty_suffstat: Fx::Stat,
    /// Cumulative CDF values are dropped after this threshold.
    cdf_threshold: f64,
    /// Point at which to stop storing runlength probabilities (prevents storing probabilities ~ 0)
    cutoff_threadhold: f64,
}

impl<X, Fx, Pr> BocpdTruncated<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx, Posterior = Pr> + Clone,
    Fx::Stat: Clone,
{
    /// Create a new Bocpd analyzer
    ///
    /// # Parameters
    /// * `hazard` - The hazard function for `P_{gap} = 1/hazard`.
    /// * `predictive_prior` - Prior for the predictive distribution.
    ///
    /// # Example
    /// ```rust
    /// use changepoint::BocpdTruncated;
    /// use rv::prelude::*;
    ///
    /// let cpd = BocpdTruncated::new(
    ///     250.0,
    ///     NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
    /// );
    /// ```
    pub fn new(hazard_lambda: f64, predictive_prior: Pr) -> Self {
        let mut rng = SmallRng::seed_from_u64(0xABCD);
        let fx: Fx = predictive_prior.draw(&mut rng);
        let empty_suffstat = fx.empty_suffstat();

        Self {
            hazard: hazard_lambda.recip(),
            predictive_prior,
            suff_stats: VecDeque::new(),
            r: Vec::new(),
            empty_suffstat,
            cdf_threshold: 1E-3,
            cutoff_threadhold: 1E-6,
            initial_suffstat: None,
        }
    }

    /// Change the cutoff for mass to be discarded on the tail end of run-lengths
    #[must_use]
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

impl<X, Fx, Pr> BocpdTruncated<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx, Posterior = Pr> + Clone,
    Fx::Stat: Clone,
{
    /// Reduce the observed values into a new BOCPD with those observed values integrated into the
    /// prior.
    #[must_use]
    pub fn collapse_stats(self) -> Self {
        let new_prior: Pr = self.suff_stats.back().map_or(
            self.predictive_prior.clone(),
            |suff_stat| {
                self.predictive_prior
                    .posterior(&DataOrSuffStat::SuffStat(suff_stat))
            },
        );

        Self {
            suff_stats: VecDeque::new(),
            r: vec![],
            predictive_prior: new_prior,
            ..self
        }
    }
}

impl<X, Fx, Pr> BocpdLike<X> for BocpdTruncated<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx, Posterior = Pr> + Clone,
    Fx::Stat: Clone,
{
    type Fx = Fx;
    type PosteriorPredictive = Mixture<Pr>;

    /// Update the model with a new datum and return the distribution of run lengths.
    fn step(&mut self, data: &X) -> &[f64] {
        if self.r.is_empty() {
            self.suff_stats.push_front(
                self.initial_suffstat
                    .clone()
                    .unwrap_or_else(|| self.empty_suffstat.clone()),
            );
            // The initial point is, by definition, a change point
            self.r.push(1.0);
        } else {
            self.suff_stats.push_front(self.empty_suffstat.clone());
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

                // Renormalize r
                let this_r_sum: f64 = self.r.iter().sum();
                for i in 0..self.r.len() {
                    self.r[i] /= this_r_sum;
                }
            }
        }

        // Update the SuffStat with the new data
        self.suff_stats
            .iter_mut()
            .for_each(|stat| stat.observe(data));

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

    fn pp(&self) -> Self::PosteriorPredictive {
        if self.suff_stats.is_empty() {
            let post = self.initial_suffstat.clone().map_or_else(
                || self.predictive_prior.clone(),
                |ss| {
                    self.predictive_prior
                        .posterior(&DataOrSuffStat::SuffStat(&ss))
                },
            );
            Mixture::uniform(vec![post])
                .expect("The mixture could not be constructed")
        } else {
            let dists: Vec<Pr::Posterior> = self
                .suff_stats
                .iter()
                .take(self.r.len())
                .map(|ss| {
                    self.predictive_prior
                        .posterior(&DataOrSuffStat::SuffStat(ss))
                })
                .collect();
            Mixture::new(self.r.clone(), dists)
                .expect("The mixture could not be constructed")
        }
    }

    fn preload(&mut self, data: &[X]) {
        let mut stat = self.empty_suffstat.clone();
        stat.observe_many(data);
        self.initial_suffstat = Some(stat);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators;
    use crate::utils::{map_changepoints, max_error};
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
            NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(),
        );

        let res: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).to_vec()).collect();

        for row in &res {
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
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
        );

        let rs: Vec<Vec<f64>> =
            data.iter().map(|d| (*cpd.step(d)).into()).collect();
        let change_points = map_changepoints(&rs);

        let error = max_error(&change_points, &[0, 499]);
        assert!(error <= 1);
    }

    #[test]
    fn coal_mining_data() {
        let data = generators::coal_mining_incidents();

        let mut cpd =
            BocpdTruncated::new(100.0, Gamma::new_unchecked(1.0, 1.0));

        let rs: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).into()).collect();
        let change_points = map_changepoints(&rs);

        let error = max_error(&change_points, &[0, 40, 95]);
        assert!(error <= 1);
    }

    /// This test checks for change points with 3-month treasury bill market data
    ///
    /// # Data Source
    /// > Board of Governors of the Federal Reserve System (US), 3-Month Treasury Bill: Secondary
    /// > Market Rate [TB3MS], retrieved from FRED, Federal Reserve Bank of St. Louis;
    /// > <https://fred.stlouisfed.org/series/TB3MS>, March 24, 2020.
    #[test]
    fn treasury_changes() -> Result<(), Box<dyn std::error::Error + 'static>> {
        let raw_data: &str = include_str!("../../resources/TB3MS.csv");
        let data: Vec<f64> = raw_data
            .lines()
            .skip(1)
            .map(|line| {
                let (_, line) = line.split_at(11);
                line.parse().unwrap()
            })
            .collect();

        let mut cpd =
            BocpdTruncated::new(250.0, NormalGamma::new(0.0, 1.0, 1.0, 1.0)?);

        let rs: Vec<Vec<f64>> = data
            .iter()
            .zip(data.iter().skip(1))
            .map(|(a, b)| (b - a) / a)
            .map(|d| cpd.step(&d).to_vec())
            .collect();

        let change_points = map_changepoints(&rs);

        let error = max_error(
            &change_points,
            &[0, 66, 93, 293, 295, 887, 898, 900, 931, 936, 977, 982],
        );
        assert!(error <= 1);
        Ok(())
    }
}
