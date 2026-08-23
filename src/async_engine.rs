//! Module 5 async isolation (Ouroboros v7.1 paper §7.3–7.4).
//!
//! LAW: equality saturation is a background CPU task on tokio's blocking
//! pool — the caller's submit/poll loop NEVER blocks on saturation, so GPU
//! inference and training loops stay hot.
//!
//! Architecture (composition over the sync engine; `engine.rs` untouched):
//! - [`AsyncEggEngine::submit`] clones the cheap [`EggEngine`] (a
//!   `Copy`-sized [`crate::limits::RunnerLimits`] inside), hands it to
//!   `tokio::task::spawn_blocking`, and returns immediately.
//! - Each worker measures its own wall time and ships
//!   `(job_id, Verdict, elapsed_ms)` through one unbounded mpsc channel.
//! - [`AsyncEggEngine::results`] drains ONLY completed jobs via
//!   `try_recv` — a non-blocking poll that returns whatever is ready,
//!   possibly nothing.
//! - Every drained job also lands as a [`VerificationRecord`] in the owned
//!   [`TelemetrySink`], drainable via [`AsyncEggEngine::take_telemetry`].
//!
//! TOKIO API SOURCES (verified against local crate source, not guessed):
//! - `~/.cargo/registry/src/*/tokio-1.53.1/src/task/blocking.rs:220`
//!     - `pub fn spawn_blocking<F, R>(f: F) -> JoinHandle<R>`
//!       where `F: FnOnce() -> R + Send + 'static, R: Send + 'static`.
//!       `EggEngine` qualifies: it is `Clone` and holds only a
//!       `RunnerLimits` (`u64`/`usize`/`Duration` → auto `Send`).
//! - `~/.cargo/registry/src/*/tokio-1.53.1/src/sync/mpsc/unbounded.rs`
//!     - `unbounded_channel<T>() -> (UnboundedSender<T>, UnboundedReceiver<T>)`
//!       (line ~95),
//!       `UnboundedSender::send(&self, T) -> Result<(), SendError<T>>`
//!       (~547; never blocks or awaits),
//!       `UnboundedReceiver::try_recv(&mut self) -> Result<T, TryRecvError>`
//!       (~286; the non-blocking drain primitive behind `results()`).
//! - `#[tokio::test]` proc-macro attribute:
//!   `~/.cargo/registry/src/*/tokio-macros-2.7.2/src/lib.rs` line ~607.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::mpsc;

use crate::engine::{EggEngine, Verdict};
use crate::telemetry::{TelemetrySink, VerificationRecord};

/// One queued verification job (owned strings cross into the worker).
struct Job {
    id: u64,
    before: String,
    after: String,
}

/// Asynchronous wrapper around the law-bounded [`EggEngine`].
///
/// Submit from any task; poll [`Self::results`] at your own cadence — each
/// call costs microseconds regardless of how deep any job is in equality
/// saturation.
#[derive(Debug)]
pub struct AsyncEggEngine {
    /// Prototype engine cloned per job; pinned to `RunnerLimits::law_mandated`.
    prototype: EggEngine,
    result_tx: mpsc::UnboundedSender<(u64, Verdict, u128)>,
    result_rx: mpsc::UnboundedReceiver<(u64, Verdict, u128)>,
    /// Jobs submitted but not yet surfaced by `results()`.
    outstanding: AtomicUsize,
    telemetry: TelemetrySink,
}

impl AsyncEggEngine {
    /// Create an engine whose workers inherit the mandated LAW limits.
    pub fn new() -> Self {
        let (result_tx, result_rx) = mpsc::unbounded_channel();
        Self {
            prototype: EggEngine::new(),
            result_tx,
            result_rx,
            outstanding: AtomicUsize::new(0),
            telemetry: TelemetrySink::new(),
        }
    }

    /// Queue one equivalence check onto the blocking pool. Returns
    /// immediately; never touches saturation state.
    ///
    /// Must be called within a tokio runtime context (`#[tokio::test]`,
    /// `#[tokio::main]`, or an explicit runtime handle), because
    /// `spawn_blocking` requires one.
    pub fn submit(&self, job_id: u64, before: String, after: String) {
        let job = Job { id: job_id, before, after };
        let engine = self.prototype.clone();
        let tx = self.result_tx.clone();
        self.outstanding.fetch_add(1, Ordering::Relaxed);

        // Fire-and-forget JoinHandle: results travel over the channel, so we
        // intentionally do not await the task itself. If the receiver side is
        // gone the send fails harmlessly (engine shutting down).
        let _ = tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let verdict = engine.verify(&job.before, &job.after);
            let elapsed_ms = started.elapsed().as_millis();
            let _ = tx.send((job.id, verdict, elapsed_ms));
        });
    }

    /// Non-blocking drain of every COMPLETED job.
    ///
    /// Never waits on running saturations: with zero finished jobs this
    /// returns an empty `Vec` in microseconds. Each drained job is recorded
    /// into the internal [`TelemetrySink`] before being returned.
    pub fn results(&mut self) -> Vec<(u64, Verdict)> {
        let mut done = Vec::new();
        while let Ok((job_id, verdict, elapsed_ms)) = self.result_rx.try_recv() {
            self.outstanding.fetch_sub(1, Ordering::Relaxed);
            self.telemetry
                .record(VerificationRecord::new(job_id, &verdict, elapsed_ms));
            done.push((job_id, verdict));
        }
        done
    }

    /// Jobs submitted but not yet drained through [`Self::results`].
    pub fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Relaxed)
    }

    /// Read-only view of accumulated SRE telemetry.
    pub fn telemetry(&self) -> &TelemetrySink {
        &self.telemetry
    }

    /// Hand the whole [`TelemetrySink`] to the owner, leaving an empty one
    /// behind.
    pub fn take_telemetry(&mut self) -> TelemetrySink {
        std::mem::take(&mut self.telemetry)
    }
}

impl Default for AsyncEggEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::VerdictKind;
    use std::collections::HashMap;
    use std::time::Duration;

    const POLL_LATENCY_BUDGET: Duration = Duration::from_millis(50);
    const COLLECT_DEADLINE: Duration = Duration::from_secs(8);

    /// Poll until every `expected` job id came back; assert EVERY individual
    /// `results()` call stayed inside [`POLL_LATENCY_BUDGET`] even while jobs
    /// are mid-saturation, and that the whole collection beat
    /// [`COLLECT_DEADLINE`]. Deterministic: tiny expressions saturate in
    /// microseconds; budgets exist only to fail loudly if the caller ever
    /// starts BLOCKING on saturation.
    async fn collect_all(
        engine: &mut AsyncEggEngine,
        expected: usize,
    ) -> HashMap<u64, Verdict> {
        let deadline = Instant::now() + COLLECT_DEADLINE;
        let mut got = HashMap::new();
        let mut worst_poll = Duration::ZERO;
        while got.len() < expected {
            assert!(
                Instant::now() < deadline,
                "saturation did not finish within {COLLECT_DEADLINE:?}"
            );
            let t0 = Instant::now();
            for (job_id, verdict) in engine.results() {
                got.insert(job_id, verdict);
            }
            let took = t0.elapsed();
            worst_poll = worst_poll.max(took);
            tokio::task::yield_now().await;
        }
        assert!(
            worst_poll < POLL_LATENCY_BUDGET,
            "results() blocked the caller: worst poll {worst_poll:?} >= {POLL_LATENCY_BUDGET:?}"
        );
        got
    }

    #[tokio::test]
    async fn equivalent_job_flows_to_green_record() {
        let mut engine = AsyncEggEngine::new();
        engine.submit(1, "(+ a b)".to_string(), "(+ b a)".to_string());
        assert_eq!(engine.outstanding(), 1);

        // Nothing may be ready instantly — but the poll itself must be cheap.
        let t0 = Instant::now();
        let early = engine.results();
        assert!(t0.elapsed() < POLL_LATENCY_BUDGET);

        let got = collect_all(&mut engine, 1).await;
        match got.get(&1).expect("job 1 must complete") {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(*iterations >= 1);
                assert!(*nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("commuted add must verify, got Unproven: {reason}")
            }
        }

        // Early poll was empty or not; either way final state is consistent.
        assert_eq!(engine.outstanding(), 0);
        assert!(early.len() <= 1);
        assert_eq!(engine.telemetry().len(), 1);
        assert_eq!(engine.telemetry().records()[0].verdict_kind(), VerdictKind::Green);
        assert_eq!(engine.telemetry().green_yellow_ratio(), Some(1.0));
    }

    #[tokio::test]
    async fn non_equivalent_job_is_yellow() {
        let mut engine = AsyncEggEngine::new();
        engine.submit(9, "(+ a b)".to_string(), "(* a b)".to_string());
        let got = collect_all(&mut engine, 1).await;
        match got.get(&9).expect("job 9 must complete") {
            Verdict::Unproven { reason } => {
                assert_eq!(reason, "saturation exhausted without merge");
            }
            Verdict::Equivalent { .. } => panic!("add vs mul cannot prove"),
        }
        assert_eq!(engine.telemetry().len(), 1);
        assert_eq!(engine.telemetry().records()[0].verdict_kind(), VerdictKind::Yellow);
        assert_eq!(engine.telemetry().green_yellow_ratio(), Some(0.0));
    }

    #[tokio::test]
    async fn ratio_reflects_two_green_one_yellow_mix() {
        let mut engine = AsyncEggEngine::new();
        engine.submit(1, "(+ a b)".to_string(), "(+ b a)".to_string());
        engine.submit(2, "(+ (+ x y) z)".to_string(), "(+ x (+ y z))".to_string());
        engine.submit(3, "(+ p q)".to_string(), "(* p q)".to_string());

        let got = collect_all(&mut engine, 3).await;
        assert_eq!(got.len(), 3);
        assert!(matches!(got[&1], Verdict::Equivalent { .. }));
        assert!(matches!(got[&2], Verdict::Equivalent { .. }));
        assert!(matches!(got[&3], Verdict::Unproven { .. }));

        let sink = engine.take_telemetry();
        assert_eq!(sink.len(), 3);
        let ratio = sink.green_yellow_ratio().expect("non-empty");
        assert!((ratio - 2.0 / 3.0).abs() < 1e-9, "ratio {ratio}");
        // Sink handed over; engine left with a fresh empty one.
        assert!(engine.telemetry().is_empty());
    }

    #[tokio::test]
    async fn poll_loop_never_blocks_while_eight_jobs_saturate() {
        const N: u64 = 8;
        let mut engine = AsyncEggEngine::new();
        for i in 0..N {
            if i % 2 == 0 {
                engine.submit(i, format!("(+ v{i} w)"), format!("(+ w v{i})"));
            } else {
                engine.submit(i, format!("(+ s{i} t)"), format!("(* s{i} t)"));
            }
        }
        assert_eq!(engine.outstanding() as u64, N);

        let got = collect_all(&mut engine, N as usize).await;
        assert_eq!(got.len() as u64, N);
        for (id, verdict) in &got {
            if id % 2 == 0 {
                assert!(matches!(verdict, Verdict::Equivalent { .. }), "job {id}");
            } else {
                assert!(matches!(verdict, Verdict::Unproven { .. }), "job {id}");
            }
        }
        assert_eq!(engine.outstanding(), 0);
        assert_eq!(engine.telemetry().len() as u64, N);

        // JSON-lines export covers all eight records exactly once.
        let text = engine.take_telemetry().to_json_lines();
        assert_eq!(text.lines().count() as u64, N);
    }

    #[tokio::test]
    async fn malformed_input_yields_yellow_not_panic() {
        let mut engine = AsyncEggEngine::new();
        engine.submit(5, "(+ a b".to_string(), "(+ b a)".to_string());
        let got = collect_all(&mut engine, 1).await;
        assert!(matches!(got[&5], Verdict::Unproven { .. }));
        assert_eq!(engine.telemetry().green_yellow_ratio(), Some(0.0));
    }
}
