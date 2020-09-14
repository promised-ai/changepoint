# Changelog

## v0.9.0
- Added helper functions for change-point inference
    - `utils::infer_changepoints` - Determine the probabilities of each point being a change-point via Monte-Carlo.
    - `utils::infer_pseudo_cmf_changepoints` - Determines the accumulated probabilities of each point being a change-point.
    - `map_changepoints` - Determines the most likely change-points.
- Added next step predictive distribution generation, `BocpdLike::pp`.
- Added a pre-load option to BocpdLike CPDs to set an initial sufficient statistic to prevent initial swings in `P(cp)`.
- Cleaned up the new API for BocpdLike structs, the `Fx` is now generated from the prior and isn't explicitly required.
- Removed BocpdLike's `votes_*` methods as they're now irrelevant.

## v0.8.1
- Implement `Clone` and `Debug` for `Bocpd` and `BocpdTruncated`
- Remove `serde_derive` dependency
- Bump patch version of dependencies
- Add changelog
