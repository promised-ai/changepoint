//! Online Bayesian Change Point Detection
//!
//! This code is derived from
//! "Bayesian Online Changepoint Detection"; Ryan Adams, David `MacKay`; arXiv:0710.3742
//! Which can be found [here](https://arxiv.org/pdf/0710.3742.pdf).

use crate::traits::BocpdLike;
use rand::{prelude::SmallRng, SeedableRng};
use rv::prelude::*;
use std::collections::VecDeque;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Online Bayesian Change Point Detection state container.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Bocpd<X, Fx, Pr>
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
    /// Current time step
    t: usize,
    /// Run-length probabilities.
    r: Vec<f64>,
    /// Reference empty suff_stats.
    empty_suffstat: Fx::Stat,
    /// Initial suff_stat
    initial_suffstat: Option<Fx::Stat>,
    /// Cumulative CDF values are dropped after this threshold.
    cdf_threshold: f64,
}

impl<X, Fx, Pr> Bocpd<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx, Posterior = Pr> + Clone,
    Fx::Stat: Clone,
{
    /// Create a new Bocpd analyzer
    ///
    /// # Parameters
    /// * `hazard` - 1/ `hazard` is the probability of the next step being a change-point.
    /// Therefore, the larger the value, the lower the prior probability is for the any point to be
    /// a change-point. Mean run-length is `lambda - 1`.
    /// * `predictive_prior` - Prior for the predictive distribution. This should match your
    /// observed data.
    ///
    /// # Example
    /// ```rust
    /// use changepoint::Bocpd;
    /// use rv::prelude::*;
    ///
    /// let cpd = Bocpd::new(
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
            t: 0,
            r: Vec::new(),
            empty_suffstat,
            cdf_threshold: 1E-3,
            initial_suffstat: None,
        }
    }

    /// Reset the introspector and replace the predictive prior
    pub fn reset_with_prior(&mut self, predictive_prior: Pr) {
        self.predictive_prior = predictive_prior;
        self.reset();
    }

    /// Reduce the observed values into a new BOCPD with those observed values integrated into the
    /// prior.
    #[must_use]
    pub fn collapse_stats(self) -> Self {
        let new_prior: Pr = self.suff_stats.back().map_or(
            self.predictive_prior.clone(),
            |suff_stat| {
                self.predictive_prior
                    .clone()
                    .posterior(&DataOrSuffStat::SuffStat(suff_stat))
            },
        );

        Self {
            suff_stats: VecDeque::new(),
            t: 0,
            r: vec![],
            predictive_prior: new_prior,
            ..self
        }
    }
}

impl<X, Fx, Pr> BocpdLike<X> for Bocpd<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx, Posterior = Pr> + Clone,
    Fx::Stat: Clone,
{
    type Fx = Fx;
    type PosteriorPredictive = Mixture<Pr>;

    fn reset(&mut self) {
        self.suff_stats.clear();
        self.t = 0;
        self.r.clear();
    }

    /// Update the model with a new datum and return the distribution of run lengths.
    fn step(&mut self, data: &X) -> &[f64] {
        if self.t == 0 {
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

            for i in (0..self.t).rev() {
                if self.r[i] == 0.0 {
                    self.r[i + 1] = 0.0;
                } else {
                    // Evaluate growth probabilities and shift probabilities down
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

        // Update total sequence length
        self.t += 1;

        &self.r
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
    use crate::utils::{infer_changepoints, map_changepoints, max_error};
    use crate::{generators, BocpdLike};
    use rand::SeedableRng;

    #[test]
    fn each_vec_is_a_probability_dist() {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 1000,
        );

        let mut cpd =
            Bocpd::new(250.0, NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap());

        let res: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).to_vec()).collect();

        for row in &res {
            let sum: f64 = row.iter().sum();
            assert::close(sum, 1.0, 1E-8);
        }
    }

    #[test]
    fn detect_obvious_switch() {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 700,
        );

        let mut cpd =
            Bocpd::new(250.0, NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0));

        let rs: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).into()).collect();
        let change_points = map_changepoints(&rs);

        let error = max_error(&change_points, &[0, 499]);
        assert!(error <= 1);
    }

    #[test]
    fn detect_obvious_switch_p_cp(
    ) -> Result<(), Box<dyn std::error::Error + 'static>> {
        let mut rng = SmallRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(
            &mut rng, 0.0, 1.0, 10.0, 5.0, 500, 700,
        );

        let mut cpd =
            Bocpd::new(250.0, NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0));

        let rs: Vec<Vec<f64>> =
            data.iter().map(|d| cpd.step(d).to_vec()).collect();
        let p_cp = infer_changepoints(&rs, 30, &mut rng)?;
        assert!(p_cp[499] > 0.5);
        Ok(())
    }

    #[test]
    fn coal_mining_data() {
        let data = generators::coal_mining_incidents();

        let mut cpd = Bocpd::new(100.0, Gamma::new_unchecked(1.0, 1.0));

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
        let mut cpd = Bocpd::new(250.0, NormalGamma::new(0.0, 1.0, 1.0, 1.0)?);

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
