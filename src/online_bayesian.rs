//! Online Bayesian Change Point Detection
//!
//! This code is derived from
//! "Bayesian Online Changepoint Detection"; Ryan Adams, David MacKay; arXiv:0710.3742
//! Which can be found [here](https://arxiv.org/pdf/0710.3742.pdf).

use rv::misc::argmax;
use rv::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

/// Online Bayesian Change Point Detection state container
pub struct OBCPD<X, H, Fx, Pr>
where
    H: Fn(usize) -> f64,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    hazard: H,
    predictive_prior: Arc<Pr>,
    suff_stats: VecDeque<Fx::Stat>,
    map_locations: Vec<usize>,
    t: usize,
    r: Vec<f64>,
    empty_suffstat: Fx::Stat,
}

impl<X, H, Fx, Pr> OBCPD<X, H, Fx, Pr>
where
    H: Fn(usize) -> f64,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    /// Create a new OBCPD analyzer
    ///
    /// # Parameters
    /// * `hazard` - The hazard function for `P_{gap}`.
    /// * `fx` - Predictive distribution. Used for generating an empty `SuffStat`.
    /// * `predictive_prior` - Prior for the predictive distribution.
    ///
    /// # Example
    /// ```rust
    /// use cpd::{OBCPD, constant_hazard};
    /// use rv::prelude::*;
    /// use std::sync::Arc;
    ///
    /// let cpd = OBCPD::new(
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
            map_locations: Vec::new(),
            t: 0,
            r: Vec::new(),
            empty_suffstat: fx.empty_suffstat(),
        }
    }

    /// Returns the MAP lengths at each step in the sequence.
    pub fn maps(&self) -> &Vec<usize> {
        &(self.map_locations)
    }

    /// Update the model with a new datum and return the distribution of run lengths.
    pub fn step(&mut self, data: &X) -> Vec<f64> {
        self.suff_stats.push_front(self.empty_suffstat.clone());

        if self.t == 0 {
            // The initial point is, by definition, a change point
            self.r.push(1.0);
        } else {
            let pred_probs: Vec<f64> = self
                .suff_stats
                .iter()
                .map(|stat| {
                    self.predictive_prior
                        .ln_pp(data, &DataOrSuffStat::SuffStat(stat))
                        .exp()
                })
                .collect();

            // println!("pp = {:?}", pred_probs);

            // evaluate the hazard function
            let h: Vec<f64> = (0..=self.t).map(|i| (self.hazard)(i)).collect();

            // TODO: The following loops could be wrapped into one.
            // TODO: There is no need to duplicate r but it made it easier to write.
            let mut new_r: Vec<f64> = Vec::with_capacity(self.t + 1);

            // Evaluate growth probabilites and shift probabilities down
            // scaling by the hazard function and the predprobs
            new_r.push(std::f64::NAN); // Place holder we'll recalc later
            for i in 0..self.t {
                new_r.push(self.r[i] * pred_probs[i] * (1.0 - h[i]));
            }

            // Accumulate mass back down to r[0], the probability there was a
            // change point at this location.
            new_r[0] = (0..self.t).map(|i| self.r[i] * pred_probs[i] * h[i]).sum();

            // Normalize R
            let new_r_sum: f64 = new_r.iter().sum();
            new_r.iter_mut().for_each(|x| {
                *x = *x / new_r_sum;
            });
            self.r = new_r;
        }

        // Update the SuffStat with the new data
        self.suff_stats
            .iter_mut()
            .for_each(|stat| stat.observe(data));

        // Update total sequence length
        self.t = self.t + 1;

        // Update MAP location
        self.map_locations.push(*argmax(&self.r).last().unwrap());
        self.r.clone()
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
    fn detect_obvious_switch() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = generators::discontinuous_jump(&mut rng, 0.0, 1.0, 10.0, 5.0, 500, 1000);

        let mut cpd = OBCPD::new(
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

        let mut cpd: OBCPD<u8, _, Poisson, Gamma> = OBCPD::new(
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

    /// This test checks the OBCPD algorithm against S&P/Case-Schiller 20-City
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

        let mut cpd = OBCPD::new(
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
