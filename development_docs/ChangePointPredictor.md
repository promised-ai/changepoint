# Change-Point Predictor Design

The change-point distributions return from the `RunLengthDetector` describe the distribution $`p(r_t | x_{0:t-1})`$ where $`r_t`$ is the run-length at some step $`t`$ and $`x_{0:t-1}`$ are the time series's values for all values up to step $`t`$.
We are most interested in the points that are likely change-points.

Let $`P(r_t = i| x_{0:t-1}) = R_{i, t}`$; we can consider the sequence $`S_n = \{ R_{n+m, m} \}_{m = 0}^{t - n - 1}`$ as describing a sequence of non-i.i.d. Bernoulli trials.
Such a distribution is described by [Poisson binomial distribution](https://en.wikipedia.org/wiki/Poisson_binomial_distribution).
The mean and variance are given below
```math
\mu = \sum_{i=0}^{n-1} p_i \\

\sigma^2 = \sum_{i=0}^{n-1} (1 - p_i) p_i
```
These values could be interpreted as the number of "votes" for a particular iteration being a change-point.
Since there is presently no way to know the number of change-points, this cannot be made into a probability.
For detecting change-points, these votes could be thresholded or, in general, some heuristic could be applied to determine if the point was a change-point.

Ideally we would use the CDF to determine a confidence interval on the number of votes but those methods are computationally more expensive.
There are non-direct methods to do this, see [On Computing the Distribution Function for the Poisson Binomial Distribution, Yili Hong](https://www.researchgate.net/profile/Yili_Hong/publication/257017356_On_computing_the_distribution_function_for_the_Poisson_binomial_distribution/links/5a02feebaca2720c32650fb3/On-computing-the-distribution-function-for-the-Poisson-binomial-distribution.pdf).

## Implementation Options

Accumulating values from the diagonal would be fairly straightforward as at each step the values could be added to a running running Vec.
This means determining the mean number of votes would be fairly easy.

This could be done in a few ways:

    1. A new method like, `RunLengthDetector::get_changepoint_votes(&self) -> Vec<f64>`
    2. A wrapper to provide a container and method, same 1 but external to the RunLengthDetector trait.

Open questions:

    1. Should there be a method to threshold values?
    2. Should we create a confidence interval estimator?


## Example

```rust
use changepoint::{Bocpd, RunLengthDetector};

fn main() -> io::Result<()> {
    
    let data: Vec<f64> = load_data();

    let mut cpd = Bocpd::new(250.0, Gaussian::standard(), NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1E-5));

    for datum in data {
        cpd.step(&datum);
    }

    let vote_vec: Vec<f64> = cpd.get_changepoint_votes();

    for (i, votes) in vote_vec.iter().enumerate() {
        println!("{}: {}", i, votes);
    }
}
```
