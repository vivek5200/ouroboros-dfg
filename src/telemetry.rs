//! Module 5 SRE telemetry (Ouroboros v7.1 paper §7.3–7.4).
//!
//! Every verification job that flows through [`crate::async_engine`] is
//! distilled into a [`VerificationRecord`] and accumulated in a
//! [`TelemetrySink`], giving operations a green/yellow health view of the
//! DFG verifier without ever touching GPU-side inference loops.
//!
//! DEPENDENCY SOURCES (verified against local crate source, not guessed):
//! - `~/.cargo/registry/src/*/serde_json-1.0.151/src/ser.rs:148`:
//!   `serialize_u128` is UNCONDITIONAL (no `arbitrary_precision` gate), so
//!   the `elapsed_ms: u128` field serializes as a plain JSON number.
//! - `~/.cargo/registry/src/*/serde_json-1.0.151/src/de.rs:388,1515`:
//!   `do_deserialize_u128` is wired into `deserialize_u128`, so the same
//!   field round-trips back into `u128`.

use serde::{Deserialize, Serialize};

use crate::engine::Verdict;

/// Traffic-light classification of an [`engine::Verdict`] (§7.4).
///
/// - `Green` ← `Verdict::Equivalent { .. }`
/// - `Yellow` ← `Verdict::Unproven { .. }` (saturation exhausted or bad input;
///   a *yellow* is a legitimate outcome, not an error).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictKind {
    /// Equivalence was proven by equality saturation.
    Green,
    /// Saturation ended without proving equivalence.
    Yellow,
}

impl From<&Verdict> for VerdictKind {
    fn from(verdict: &Verdict) -> Self {
        match verdict {
            Verdict::Equivalent { .. } => VerdictKind::Green,
            Verdict::Unproven { .. } => VerdictKind::Yellow,
        }
    }
}

/// One completed verification job, ready for SRE aggregation.
///
/// Field types are pinned by the module contract (`job_id: u64`,
/// `verdict_kind: VerdictKind`, `iterations/nodes: usize`,
/// `elapsed_ms: u128`). For `Yellow` verdicts `iterations`/`nodes` are `0`
/// because `Verdict::Unproven` carries only a reason string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationRecord {
    job_id: u64,
    verdict_kind: VerdictKind,
    iterations: usize,
    nodes: usize,
    elapsed_ms: u128,
}

impl VerificationRecord {
    /// Build a record from a finished job. `elapsed_ms` is measured by the
    /// caller around the saturation call (see [`crate::async_engine`]).
    pub fn new(job_id: u64, verdict: &Verdict, elapsed_ms: u128) -> Self {
        let verdict_kind = VerdictKind::from(verdict);
        let (iterations, nodes) = match verdict {
            Verdict::Equivalent { iterations, nodes } => (*iterations, *nodes),
            Verdict::Unproven { .. } => (0, 0),
        };
        Self {
            job_id,
            verdict_kind,
            iterations,
            nodes,
            elapsed_ms,
        }
    }

    pub fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn verdict_kind(&self) -> VerdictKind {
        self.verdict_kind
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn nodes(&self) -> usize {
        self.nodes
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed_ms
    }
}

/// Accumulator of [`VerificationRecord`]s with aggregate health views.
#[derive(Debug, Clone, Default)]
pub struct TelemetrySink {
    records: Vec<VerificationRecord>,
}

impl TelemetrySink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one completed-job record.
    pub fn record(&mut self, record: VerificationRecord) {
        self.records.push(record);
    }

    /// Number of accumulated records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Read-only view of the accumulated records (arrival order).
    pub fn records(&self) -> &[VerificationRecord] {
        &self.records
    }

    /// `greens / (greens + yellows)`; [`None`] when no records exist.
    ///
    /// Only-yellow history → `Some(0.0)`; only-green → `Some(1.0)`.
    pub fn green_yellow_ratio(&self) -> Option<f64> {
        if self.records.is_empty() {
            return None;
        }
        let greens = self
            .records
            .iter()
            .filter(|r| r.verdict_kind == VerdictKind::Green)
            .count();
        Some(greens as f64 / self.records.len() as f64)
    }

    /// Serialize as JSON Lines: exactly one JSON object per line, newline
    /// terminated (empty sink → empty string).
    ///
    /// `u128` support verified against local serde_json-1.0.151 sources
    /// (module docs). Panics only if serialization of our own plain-data
    /// records fails, which cannot happen for `Vec`/integers/enums.
    pub fn to_json_lines(&self) -> String {
        let mut out = String::new();
        for record in &self.records {
            out.push_str(
                &serde_json::to_string(record)
                    .expect("VerificationRecord is pure data; serialization is infallible"),
            );
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    /// Test-only mirror used to prove `to_json_lines` output parses back
    /// through `serde_json::from_str` with identical shape and values.
    #[derive(Debug, Deserialize, PartialEq)]
    struct RecordMirror {
        job_id: u64,
        verdict_kind: VerdictKind,
        iterations: usize,
        nodes: usize,
        elapsed_ms: u128,
    }

    fn green(id: u64, iterations: usize, nodes: usize, ms: u128) -> VerificationRecord {
        VerificationRecord::new(
            id,
            &Verdict::Equivalent { iterations, nodes },
            ms,
        )
    }

    #[test]
    fn equivalent_maps_to_green_with_metrics() {
        let rec = green(7, 12, 340, 5);
        assert_eq!(rec.verdict_kind(), VerdictKind::Green);
        assert_eq!(rec.job_id(), 7);
        assert_eq!(rec.iterations(), 12);
        assert_eq!(rec.nodes(), 340);
        assert_eq!(rec.elapsed_ms(), 5);
    }

    #[test]
    fn unproven_maps_to_yellow_without_metrics() {
        let rec = VerificationRecord::new(
            3,
            &Verdict::Unproven { reason: "no merge".into() },
            42,
        );
        assert_eq!(rec.verdict_kind(), VerdictKind::Yellow);
        assert_eq!(rec.iterations(), 0);
        assert_eq!(rec.nodes(), 0);
        assert_eq!(rec.elapsed_ms(), 42);
    }

    #[test]
    fn two_green_one_yellow_ratio_is_two_thirds() {
        let mut sink = TelemetrySink::new();
        sink.record(green(1, 3, 30, 1));
        sink.record(green(2, 4, 40, 1));
        sink.record(VerificationRecord::new(
            3,
            &Verdict::Unproven { reason: "x".into() },
            2,
        ));
        let ratio = sink.green_yellow_ratio().expect("non-empty");
        assert!((ratio - 2.0 / 3.0).abs() < 1e-9, "got {ratio}");
    }

    #[test]
    fn empty_sink_ratio_is_none_and_jsonl_is_empty() {
        let sink = TelemetrySink::new();
        assert!(sink.is_empty());
        assert_eq!(sink.green_yellow_ratio(), None);
        assert_eq!(sink.to_json_lines(), "");
    }

    #[test]
    fn degenerate_ratios_are_zero_and_one() {
        let mut yellows_only = TelemetrySink::new();
        yellows_only.record(VerificationRecord::new(
            1,
            &Verdict::Unproven { reason: "y".into() },
            1,
        ));
        assert_eq!(yellows_only.green_yellow_ratio(), Some(0.0));

        let mut greens_only = TelemetrySink::new();
        greens_only.record(green(1, 1, 1, 1));
        assert_eq!(greens_only.green_yellow_ratio(), Some(1.0));
    }

    #[test]
    fn json_lines_round_trip_through_mirror_struct() {
        let mut sink = TelemetrySink::new();
        sink.record(green(11, 9, 210, 17));
        sink.record(VerificationRecord::new(
            12,
            &Verdict::Unproven { reason: "stuck".into() },
            250,
        ));

        let text = sink.to_json_lines();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: RecordMirror =
            serde_json::from_str(lines[0]).expect("line 1 must parse");
        assert_eq!(
            first,
            RecordMirror {
                job_id: 11,
                verdict_kind: VerdictKind::Green,
                iterations: 9,
                nodes: 210,
                elapsed_ms: 17,
            }
        );

        let second: RecordMirror =
            serde_json::from_str(lines[1]).expect("line 2 must parse");
        assert_eq!(second.verdict_kind, VerdictKind::Yellow);
        assert_eq!(second.job_id, 12);
        assert_eq!(second.elapsed_ms, 250);
    }
}
