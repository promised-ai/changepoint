//! A demo of the online Bayesian change point detection on
//! the 3-month Treasury Bill Secondary Market Rate from
//!
//! After this example is run, the file `trasury_bill.ipynb` can be run to generate
//! plots for this dataset.
//!
//! > Board of Governors of the Federal Reserve System (US), 3-Month Treasury Bill: Secondary
//! > Market Rate [TB3MS], retrieved from FRED, Federal Reserve Bank of St. Louis;
//! > https://fred.stlouisfed.org/series/TB3MS, August 5, 2019.

use changepoint::{constant_hazard, utils, BocpdTruncated, MapPathDetector};
use rv::prelude::*;
use std::io;

fn main() -> io::Result<()> {
    // Parse the data from the TB3MS dataset
    let data: &str = include_str!("../resources/TB3MS.csv");
    let (dates, pct_change): (Vec<&str>, Vec<f64>) = data
        .lines()
        .skip(1)
        .map(|line| {
            let split: Vec<&str> = line.splitn(2, ',').collect();
            let date = split[0];
            let raw_pct = split[1];
            (date, raw_pct.parse::<f64>().unwrap())
        })
        .unzip();

    // Create the Bocpd processor
    let mut cpd = utils::MostLikelyPathWrapper::new(BocpdTruncated::new(
        constant_hazard(250.0),
        Gaussian::standard(),
        NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1E-5),
    ));

    // Feed data into change point detector
    let res: Vec<Vec<f64>> = pct_change
        .iter()
        .map(|d| cpd.step(d).map_path_probs.clone().into())
        .collect();

    // Determine most likely change points
    let change_points: Vec<usize> =
        utils::ChangePointDetectionMethod::DropThreshold(0.1).detect(&res);
    let change_dates: Vec<&str> =
        change_points.iter().map(|&i| dates[i]).collect();

    // Write output for processing my `trasury_bill.ipynb`.
    utils::write_data_and_r(
        "treasury_bill_output",
        &pct_change,
        &res,
        &change_points,
    )?;

    println!("Most likely dates of changes = {:#?}", change_dates);

    Ok(())
}
