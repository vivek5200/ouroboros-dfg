//! Hardcoded egg::Runner resource limits (egraph-limits law).
//!
//! These values are MANDATED by `memory_seeds/laws.json` (confidence 0.99)
//! to prevent exponential RAM exhaustion and undecidability bottlenecks
//! (Rice's Theorem). The unit tests in this module pin the exact numbers —
//! changing a constant without updating the law is a build failure.

use std::fmt;
use std::time::Duration;

/// Maximum e-graph rewrite iterations.
pub const ITERATION_LIMIT: u64 = 5000;
/// Maximum wall-clock time for one equality-saturation run.
pub const TIME_LIMIT: Duration = Duration::from_secs(10);
/// Maximum e-graph node count.
pub const NODE_LIMIT: u64 = 1_000_000;
/// BackoffScheduler: matches per rule before the rule is banned.
pub const BACKOFF_MATCH_LIMIT: usize = 5000;
/// BackoffScheduler: iterations a banned rule stays banned.
pub const BACKOFF_BAN_LENGTH: usize = 3;

/// The complete, law-mandated resource envelope for any `egg::Runner`.
///
/// Construct engines exclusively through this struct so the limits cannot
/// be bypassed or partially applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunnerLimits {
    pub iteration_limit: u64,
    pub time_limit: Duration,
    pub node_limit: u64,
    pub backoff_match_limit: usize,
    pub backoff_ban_length: usize,
}

impl Default for RunnerLimits {
    fn default() -> Self {
        Self::law_mandated()
    }
}

impl RunnerLimits {
    /// The single lawful configuration. No other instance should exist.
    pub const fn law_mandated() -> Self {
        Self {
            iteration_limit: ITERATION_LIMIT,
            time_limit: TIME_LIMIT,
            node_limit: NODE_LIMIT,
            backoff_match_limit: BACKOFF_MATCH_LIMIT,
            backoff_ban_length: BACKOFF_BAN_LENGTH,
        }
    }

    /// Panics if any field drifted from the law — used as a runtime guard
    /// right before engine construction.
    pub fn assert_lawful(&self) {
        assert_eq!(self.iteration_limit, ITERATION_LIMIT, "iteration limit drifted from law");
        assert_eq!(self.time_limit, TIME_LIMIT, "time limit drifted from law");
        assert_eq!(self.node_limit, NODE_LIMIT, "node limit drifted from law");
        assert_eq!(
            self.backoff_match_limit, BACKOFF_MATCH_LIMIT,
            "backoff match_limit drifted from law"
        );
        assert_eq!(
            self.backoff_ban_length, BACKOFF_BAN_LENGTH,
            "backoff ban_length drifted from law"
        );
    }
}

impl fmt::Display for RunnerLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RunnerLimits(iterations={}, time={}s, nodes={}, backoff(match={}, ban={}))",
            self.iteration_limit,
            self.time_limit.as_secs(),
            self.node_limit,
            self.backoff_match_limit,
            self.backoff_ban_length
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LAW PIN: these exact numbers come from memory_seeds/laws.json
    /// (egraph-limits, confidence 0.99). Changing one is a law violation.
    #[test]
    fn constants_match_architectural_law() {
        assert_eq!(ITERATION_LIMIT, 5000);
        assert_eq!(TIME_LIMIT, Duration::from_secs(10));
        assert_eq!(NODE_LIMIT, 1_000_000);
        assert_eq!(BACKOFF_MATCH_LIMIT, 5000);
        assert_eq!(BACKOFF_BAN_LENGTH, 3);
    }

    #[test]
    fn law_mandated_config_matches_constants() {
        let l = RunnerLimits::law_mandated();
        assert_eq!(l.iteration_limit, ITERATION_LIMIT);
        assert_eq!(l.time_limit, TIME_LIMIT);
        assert_eq!(l.node_limit, NODE_LIMIT);
        assert_eq!(l.backoff_match_limit, BACKOFF_MATCH_LIMIT);
        assert_eq!(l.backoff_ban_length, BACKOFF_BAN_LENGTH);
    }

    #[test]
    fn assert_lawful_accepts_mandated_config() {
        RunnerLimits::law_mandated().assert_lawful();
    }

    #[test]
    #[should_panic(expected = "iteration limit drifted")]
    fn assert_lawful_rejects_drift() {
        let mut l = RunnerLimits::law_mandated();
        l.iteration_limit = 999_999;
        l.assert_lawful();
    }

    #[test]
    fn display_contains_all_bounds() {
        let s = RunnerLimits::law_mandated().to_string();
        for needle in ["5000", "10s", "1000000", "ban=3"] {
            assert!(s.contains(needle), "display missing {needle}: {s}");
        }
    }
}
