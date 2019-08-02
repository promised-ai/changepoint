use rand::Rng;
use rv::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

pub struct BCPD<X, H, Fx, Pr>
where
    H: Fn(usize) -> f64,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    hazard: H,
    predictive_prior: Arc<Pr>,
    suff_stats: Vec<Fx::Stat>,
    map_locations: Vec<usize>,
    t: usize,
    r: Vec<f64>,
    empty_suffstat: Fx::Stat,
}

impl<X, H, Fx, Pr> BCPD<X, H, Fx, Pr>
where
    H: Fn(usize) -> f64,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    pub fn new(hazard: H, fx: &Fx, predictive_prior: Arc<Pr>) -> Self {
        Self {
            hazard,
            predictive_prior,
            suff_stats: Vec::new(),
            map_locations: Vec::new(),
            t: 0,
            r: Vec::new(),
            empty_suffstat: fx.empty_suffstat(),
        }
    }

    pub fn maps(&self) -> &Vec<usize> {
        &(self.map_locations)
    }

    pub fn step(&mut self, data: &X) -> Vec<f64> {
        if self.t == 0 {
            // The initial point is, by definition, a change point
            let mut next_suffstat = self.empty_suffstat.clone();
            next_suffstat.observe(data);
            self.suff_stats.push(next_suffstat);
            self.r.push(1.0);
        } else {
            let pred_probs: Vec<f64> = self.suff_stats.iter().map(|stat| {
                self.predictive_prior.ln_pp(data, &DataOrSuffStat::SuffStat(stat)).exp()
            }).collect();

            // evaluate the hazard function
            let h: Vec<f64> = (0..=self.t).map(|i| (self.hazard)(i)).collect();

            // Extend R by one
            // self.r.push(0.0);
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

            // Update the SuffStat with the new data
            let mut next_suffstat = self.suff_stats.last().unwrap().clone();
            next_suffstat.observe(data);
            self.suff_stats.push(next_suffstat);
        }

        // Update struct
        self.t = self.t + 1;

        // Update MAP location
        self.map_locations.push(Self::argmax(&self.r));
        self.r.clone()
    }

    fn argmax(xs: &[f64]) -> usize {
        xs
            .iter()
            .enumerate()
            .fold((0, std::f64::NEG_INFINITY), |(argmax, m), (i, x)| {
                match m.partial_cmp(x) {
                    Some(Ordering::Equal) => (argmax, m),
                    Some(Ordering::Greater) => (argmax, m),
                    Some(Ordering::Less) => (i, *x),
                    None => {
                        println!("Problem value = {:?}", x);
                        unreachable!();
                    }
                }
            })
            .0
    }
}

/// A constant hazard function.
pub fn constant_hazard(lambda: f64) -> impl Fn(usize) -> f64 {
    let inv_lambda = 1.0 / lambda;
    move |_: usize| inv_lambda
}

/// Generate a series of draws from two gaussian process that switches at `switch` into the sequence
pub fn discontinuous_jump<R: Rng>(
    rng: &mut R,
    mu_1: f64,
    sigma_1: f64,
    mu_2: f64,
    sigma_2: f64,
    switch: usize,
    size: usize,
) -> Vec<f64> {
    let g1 = Gaussian::new(mu_1, sigma_1).unwrap();
    let g2 = Gaussian::new(mu_2, sigma_2).unwrap();
    [g1.sample(switch, rng), g2.sample(size - switch, rng)].concat()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::Arc;
    use std::io::prelude::*;
    use std::fs::File;

    #[test]
    fn detect_obvious_switch() {
        let mut rng: StdRng = StdRng::seed_from_u64(0xABCD);
        let data = discontinuous_jump(&mut rng, 0.0, 1.0, 10.0, 3.0, 500, 1000);

        let mut cpd = BCPD::new(
            constant_hazard(250.0),
            &Gaussian::standard(),
            Arc::new(NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap()),
        );

        let res: Vec<Vec<f64>> = data.iter().map(|d| cpd.step(d)).collect();

        // Write output
        let mut f = File::create("./data.txt").unwrap();
        data.iter().for_each(|d| {writeln!(f, "{}", d).unwrap(); });

        let mut f = File::create("./R.txt").unwrap();
        let t = res.len();
        res.iter()
            .for_each(|ps| {
                for i in 0..ps.len() {
                    write!(f, "{} ", ps[i]).unwrap();
                }
                for _i in ps.len()..t {
                    write!(f, "{} ", std::f64::EPSILON).unwrap();
                }
                writeln!(f, "").unwrap();
            });
    }
}
