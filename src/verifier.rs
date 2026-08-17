//! Semantic equivalence verifier using egg equality saturation.
//!
//! CONSTRAINT: egg::Runner limits MUST be hardcoded:
//!   - IterationLimit(5000)
//!   - TimeLimit(Duration::from_secs(10))
//!   - NodeLimit(1_000_000)
//!   - BackoffScheduler { match_limit: 5000, ban_length: 3 }
//!
//! CONSTRAINT: Equality saturation MUST run as an async background CPU task.
//!   It must NEVER block the GPU inference/training loop.

use std::time::Duration;

/// Resource limits for the egg::Runner.
///
/// These are hardcoded per the Ouroboros v7.1 specification
/// to prevent exponential RAM exhaustion.
pub const ITERATION_LIMIT: usize = 5000;
pub const TIME_LIMIT_SECS: u64 = 10;
pub const NODE_LIMIT: usize = 1_000_000;
pub const MATCH_LIMIT: usize = 5000;
pub const BAN_LENGTH: usize = 3;

/// Configuration for the equality saturation verifier.
pub struct VerifierConfig {
    pub iteration_limit: usize,
    pub time_limit: Duration,
    pub node_limit: usize,
    pub match_limit: usize,
    pub ban_length: usize,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            iteration_limit: ITERATION_LIMIT,
            time_limit: Duration::from_secs(TIME_LIMIT_SECS),
            node_limit: NODE_LIMIT,
            match_limit: MATCH_LIMIT,
            ban_length: BAN_LENGTH,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_matches_spec() {
        let config = VerifierConfig::default();
        assert_eq!(config.iteration_limit, 5000);
        assert_eq!(config.time_limit, Duration::from_secs(10));
        assert_eq!(config.node_limit, 1_000_000);
        assert_eq!(config.match_limit, 5000);
        assert_eq!(config.ban_length, 3);
    }

    #[test]
    fn test_constants_match_spec() {
        assert_eq!(ITERATION_LIMIT, 5000);
        assert_eq!(TIME_LIMIT_SECS, 10);
        assert_eq!(NODE_LIMIT, 1_000_000);
        assert_eq!(MATCH_LIMIT, 5000);
        assert_eq!(BAN_LENGTH, 3);
    }
}
