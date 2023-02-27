//! Functions to generate random sequences
use rand::Rng;
use rv::dist::Gaussian;
use rv::traits::Rv;

/// Generate a series of draws from two Gaussian process that switches
/// at `switch` into the sequence.
///
/// # Example
/// ```rust
/// use changepoint::generators::discontinuous_jump;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
/// let mut rng: StdRng = StdRng::seed_from_u64(0x12345);
/// // Generate a sequence of 1000 numbers from two Gaussian, G(0, 1) and G(10, 5),
/// // switching from the first to the second at 500 steps.
/// let seq: Vec<f64> = discontinuous_jump(
///     &mut rng,
///     0.0,
///     1.0,
///     10.0,
///     5.0,
///     500,
///     1000
/// );
/// assert_eq!(seq.len(), 1000);
/// ```
pub fn discontinuous_jump<R: Rng>(
    rng: &mut R,
    mu_1: f64,
    sigma_1: f64,
    mu_2: f64,
    sigma_2: f64,
    switch: usize,
    size: usize,
) -> Vec<f64> {
    let g1 = Gaussian::new(mu_1, sigma_1).expect("Arguments should be valid");
    let g2 = Gaussian::new(mu_2, sigma_2).expect("Arguments should be valid");
    [g1.sample(switch, rng), g2.sample(size - switch, rng)].concat()
}

/// Return the coal mining disasters dataset.
///
/// From: R. G. Jarrett. A note on the intervals between coal-mining disasters.
/// Biometrika, 66(1):191–193,1979
#[must_use]
pub fn coal_mining_incidents() -> Vec<u8> {
    // Number of mining explosions in the UK from 1851 to 1962, by year.
    let data: [u8; 111] = [
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4,
        4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2,
        1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    ];
    data.to_vec()
}
