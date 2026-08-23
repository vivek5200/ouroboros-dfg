//! Module 5.1: synchronous, law-bounded equivalence checker.
//!
//! Wraps [`egg::Runner`] equality saturation behind [`EggEngine::verify`],
//! configured EXCLUSIVELY through [`crate::limits::RunnerLimits::law_mandated`]
//! per the egraph-limits LAW (see crate docs).
//!
//! EGG API SOURCES (verified against local crate source, not guessed):
//! - `~/.cargo/registry/src/*/egg-0.9.5/src/run.rs`
//!     - `Runner::new(analysis)` (line ~315; defaults iter 30 / nodes 10_000 /
//!       time 5s — all overridden below),
//!       `.with_iter_limit(usize)` (~333), `.with_node_limit(usize)` (~338),
//!       `.with_time_limit(Duration)` (~343), `.with_scheduler(...)` (~381),
//!       `.with_expr(&RecExpr<L>)` (~391, records root id in `runner.roots`),
//!       `.run(impl IntoIterator<Item = &Rewrite>)` (~406).
//!     - `BackoffScheduler` lives in `run.rs` here (NOT a `scheduler/`
//!       submodule): `::default()` (~809, defaults match_limit 1000 /
//!       ban_length 5), `.with_initial_match_limit(usize)` (~764),
//!       `.with_ban_length(usize)` (~771).
//! - `~/.cargo/registry/src/*/egg-0.9.5/src/egraph.rs`
//!     - `EGraph::equivs(&expr1, &expr2) -> Vec<Id>` (~765): the clean,
//!       non-mutating equality probe — searches the saturated e-graph for an
//!       e-class representing both expressions. No manual `add_expr` +
//!       `find` comparison needed.
//! - `~/.cargo/registry/src/*/egg-0.9.5/src/macros.rs`
//!     - `macro_rules! rewrite` (~282): the `<=>` arm (~295–303) expands to
//!       a `vec!` of TWO rewrites (`name` and `name + "-rev"`), which is why
//!       [`crate::rules`] extends bidirectional pairs into the rule vector.
//!
//! RULE SET: [`EggEngine::verify`] saturates under the UNION of the math and
//! boolean rewrite systems ([`crate::rules::all_rules`], paper §7.2 — De
//! Morgan's Laws, double-negation, algebraic identities). The families are
//! root-symbol-disjoint, so merged saturation cannot cross-prove between
//! them; see [`crate::rules::all_rules`] docs.
//!
//! TELEMETRY: [`EggEngine::verify_telemetry`] wraps [`Self::verify`] with
//! wall-clock measurement and composes the verdict into a
//! [`crate::telemetry::VerificationRecord`] via the existing
//! `VerificationRecord::new` mapping (Verdict → VerdictKind, §7.4) — no sink
//! or mapping logic is duplicated here; sinks stay in
//! [`crate::telemetry::TelemetrySink`].
//!
//! SSA ENTRY POINTS ([`Self::verify_ssa`], [`Self::verify_ssa_roots`],
//! [`Self::verify_ssa_telemetry`]): graphs lowered by
//! [`crate::ssa_bridge::to_rec_expr`] flow through the SAME LAW-bounded
//! runner pattern (assert_lawful → limit-wired Runner → `equivs`, the body
//! of [`Self::verify`] below). They saturate under
//! [`crate::rules::ssa_all_rules`] — the IR-spelled twin of the math system
//! (`add`/`mul` vs `+`/`*`; SymbolLang operators are opaque symbols, so
//! `(add ?a 0)` can never match a `+`-dialect pattern) unioned with the
//! boolean family. The classification tail is shared verbatim with
//! [`Self::verify`] through [`classify`], so there is exactly ONE definition
//! of "Equivalent" in this module.
//!
//! COMPROMISE NOTE (documented per task instructions): `RunnerLimits` stores
//! iteration/node limits as `u64` (law-pinned constants), while egg 0.9.5's
//! builder takes `usize` (run.rs lines 333/338). The casts below are lossless
//! on every target this crate builds for (`NODE_LIMIT` = 1e6 fits `usize`).

use std::time::Instant;

use egg::{BackoffScheduler, RecExpr, Runner, SymbolLang};

use crate::limits::RunnerLimits;
use crate::ssa::{Ssa, Value};

/// Outcome of one law-bounded equivalence check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Both sides share an e-class after saturation.
    Equivalent {
        /// Rewriting iterations actually performed by the runner
        /// (`runner.iterations.len()`).
        iterations: usize,
        /// E-graph enodes at the final recorded iteration
        /// (`Iteration::egraph_nodes`, run.rs ~280).
        nodes: usize,
    },
    /// Saturation ended with the two expressions in distinct e-classes
    /// (or an input failed to parse).
    Unproven { reason: String },
}

/// Synchronous equivalence checker bounded by the mandated LAW limits.
///
/// Every `Runner` built here goes through [`Self::verify`], which applies
/// `self.limits` wholesale — there is no way to construct an unbounded run.
#[derive(Debug, Clone)]
pub struct EggEngine {
    limits: RunnerLimits,
}

/// Shared classification tail of every law-bounded run (formerly the inline
/// body of [`EggEngine::verify`] at engine.rs:123–136): Equivalent iff both
/// expressions are represented by ONE e-class in the saturated graph —
/// `egg::EGraph::equivs`, egraph.rs:765–784, a NON-mutating Pattern search
/// over both trees (the `after` tree must have been materialized by
/// rewriting). Metrics: real runner iterations (`runner.iterations.len()`)
/// and the final recorded `Iteration::egraph_nodes` (run.rs:277–282), with
/// the total-node fallback for a run that recorded no iterations.
fn classify(
    runner: &Runner<SymbolLang, ()>,
    before_expr: &RecExpr<SymbolLang>,
    after_expr: &RecExpr<SymbolLang>,
) -> Verdict {
    if !runner.egraph.equivs(before_expr, after_expr).is_empty() {
        Verdict::Equivalent {
            iterations: runner.iterations.len(),
            nodes: runner
                .iterations
                .last()
                .map(|it| it.egraph_nodes)
                .unwrap_or_else(|| runner.egraph.total_number_of_nodes()),
        }
    } else {
        Verdict::Unproven {
            reason: "saturation exhausted without merge".to_string(),
        }
    }
}

impl EggEngine {
    /// Create an engine pinned to [`RunnerLimits::law_mandated`].
    pub fn new() -> Self {
        Self {
            limits: RunnerLimits::law_mandated(),
        }
    }

    /// Read-only access to the stored limits (used by tests to pin the LAW).
    pub fn limits(&self) -> &RunnerLimits {
        &self.limits
    }

    /// Check whether `before` and `after` are equivalent under the merged
    /// math + boolean rewrite system ([`crate::rules::all_rules`]), within
    /// the LAW resource envelope. Equivalence is proven iff both roots
    /// collapse into one e-class (paper §7.2).
    pub fn verify(&self, before: &str, after: &str) -> Verdict {
        let before_expr: RecExpr<SymbolLang> = match before.parse() {
            Ok(expr) => expr,
            Err(err) => return Verdict::Unproven { reason: format!("bad `before` s-expression: {err}") },
        };
        let after_expr: RecExpr<SymbolLang> = match after.parse() {
            Ok(expr) => expr,
            Err(err) => return Verdict::Unproven { reason: format!("bad `after` s-expression: {err}") },
        };

        // Runtime guard right before construction, per limits.rs contract.
        self.limits.assert_lawful();

        let rules = crate::rules::all_rules();

        let runner = Runner::<SymbolLang, ()>::new(())
            .with_iter_limit(self.limits.iteration_limit as usize) // see COMPROMISE NOTE
            .with_node_limit(self.limits.node_limit as usize) // see COMPROMISE NOTE
            .with_time_limit(self.limits.time_limit)
            .with_scheduler(
                BackoffScheduler::default()
                    .with_initial_match_limit(self.limits.backoff_match_limit)
                    .with_ban_length(self.limits.backoff_ban_length),
            )
            .with_expr(&before_expr)
            .run(&rules);

        classify(&runner, &before_expr, &after_expr)
    }

    /// Run [`Self::verify`] with wall-clock measurement and distill the
    /// outcome into a SRE [`crate::telemetry::VerificationRecord`] (§7.3–7.4).
    ///
    /// Composition, not duplication: elapsed time is the ONLY thing measured
    /// here (`std::time::Instant`); the Verdict → VerdictKind mapping and
    /// metrics extraction are delegated to the existing
    /// `VerificationRecord::new`, and sink accumulation stays with
    /// [`crate::telemetry::TelemetrySink`] — callers decide whether/where to
    /// record.
    pub fn verify_telemetry(
        &self,
        job_id: u64,
        before: &str,
        after: &str,
    ) -> crate::telemetry::VerificationRecord {
        let started = Instant::now();
        let verdict = self.verify(before, after);
        let elapsed_ms = started.elapsed().as_millis();
        crate::telemetry::VerificationRecord::new(job_id, &verdict, elapsed_ms)
    }

    /// SSA-level equivalence check with explicit roots on BOTH graphs.
    ///
    /// Converts `root_before` / `root_after` via
    /// [`crate::ssa_bridge::to_rec_expr`], then replays the EXACT pattern of
    /// [`Self::verify`] (engine.rs:106–137 pre-refactor): runtime LAW guard
    /// → [`crate::rules::ssa_all_rules`] → `Runner` wired from
    /// `self.limits` (iter/node/time + BackoffScheduler) seeded with
    /// `before`'s expression → shared [`classify`] tail on
    /// `EGraph::equivs`. Only `before`'s expression is added to the e-graph;
    /// `after` must be *materialized by rewriting* for `equivs` to find it
    /// (egraph.rs:765–784 is a non-mutating pattern search).
    pub fn verify_ssa_roots(
        &self,
        before: &Ssa,
        root_before: Value,
        after: &Ssa,
        root_after: Value,
    ) -> Verdict {
        let before_expr = crate::ssa_bridge::to_rec_expr(before, root_before);
        let after_expr = crate::ssa_bridge::to_rec_expr(after, root_after);

        // Runtime guard right before construction, per limits.rs contract —
        // identical position as in `verify`.
        self.limits.assert_lawful();

        let rules = crate::rules::ssa_all_rules();

        let runner = Runner::<SymbolLang, ()>::new(())
            .with_iter_limit(self.limits.iteration_limit as usize) // see COMPROMISE NOTE
            .with_node_limit(self.limits.node_limit as usize) // see COMPROMISE NOTE
            .with_time_limit(self.limits.time_limit)
            .with_scheduler(
                BackoffScheduler::default()
                    .with_initial_match_limit(self.limits.backoff_match_limit)
                    .with_ban_length(self.limits.backoff_ban_length),
            )
            .with_expr(&before_expr)
            .run(&rules);

        classify(&runner, &before_expr, &after_expr)
    }

    /// SSA-level equivalence: `before` vs `after` with `out` naming the
    /// result value of the `after` graph. `before`'s root is inferred as its
    /// highest defined value ([`crate::ssa_bridge::last_defined`]) — the
    /// natural output of builder-order code; use [`Self::verify_ssa_roots`]
    /// when `before`'s output is not its last def. A `before` with no
    /// defined operations is `Unproven`, never a panic.
    pub fn verify_ssa(&self, before: &Ssa, after: &Ssa, out: Value) -> Verdict {
        match crate::ssa_bridge::last_defined(before) {
            Some(root_before) => self.verify_ssa_roots(before, root_before, after, out),
            None => Verdict::Unproven {
                reason: "`before` defines no operations; nothing to saturate".to_string(),
            },
        }
    }

    /// Run [`Self::verify_ssa`] with wall-clock measurement and distill the
    /// outcome into a [`crate::telemetry::VerificationRecord`] — composition
    /// identical to [`Self::verify_telemetry`]: only elapsed time is measured
    /// here; Verdict → VerdictKind mapping and metrics extraction stay in
    /// `VerificationRecord::new`, sinks stay with `TelemetrySink`.
    pub fn verify_ssa_telemetry(
        &self,
        job_id: u64,
        before: &Ssa,
        after: &Ssa,
        out: Value,
    ) -> crate::telemetry::VerificationRecord {
        let started = Instant::now();
        let verdict = self.verify_ssa(before, after, out);
        let elapsed_ms = started.elapsed().as_millis();
        crate::telemetry::VerificationRecord::new(job_id, &verdict, elapsed_ms)
    }
}

impl Default for EggEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commuted_addition_is_equivalent() {
        let engine = EggEngine::new();
        match engine.verify("(+ a b)", "(+ b a)") {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => panic!("expected Equivalent, got Unproven: {reason}"),
        }
    }

    #[test]
    fn multiplication_is_unproven() {
        let engine = EggEngine::new();
        match engine.verify("(+ a b)", "(* a b)") {
            Verdict::Unproven { reason } => {
                assert_eq!(reason, "saturation exhausted without merge");
            }
            Verdict::Equivalent { .. } => panic!("add vs mul must NOT be provable"),
        }
    }

    #[test]
    fn engine_stores_law_mandated_limits() {
        let engine = EggEngine::new();
        assert_eq!(engine.limits(), &RunnerLimits::law_mandated());
        // Belt-and-braces: every individual field still equals its constant.
        engine.limits().assert_lawful();
        assert_eq!(engine.limits().iteration_limit, crate::limits::ITERATION_LIMIT);
        assert_eq!(engine.limits().time_limit, crate::limits::TIME_LIMIT);
        assert_eq!(engine.limits().node_limit, crate::limits::NODE_LIMIT);
        assert_eq!(engine.limits().backoff_match_limit, crate::limits::BACKOFF_MATCH_LIMIT);
        assert_eq!(engine.limits().backoff_ban_length, crate::limits::BACKOFF_BAN_LENGTH);
    }

    #[test]
    fn malformed_input_is_unproven_not_panic() {
        let engine = EggEngine::new();
        match engine.verify("(+ a b", "(+ b a)") {
            Verdict::Unproven { reason } => assert!(reason.contains("bad `before`")),
            Verdict::Equivalent { .. } => panic!("malformed input cannot prove equivalence"),
        }
    }

    // ---- Boolean rule system (paper §7.2, bidirectional rewrites) ----

    /// De Morgan #1: not(and(p,q)) ≡ or(not(p), not(q)) — proven iff both
    /// roots collapse into one e-class under bidirectional saturation.
    #[test]
    fn de_morgan_1_proves_equivalent() {
        let engine = EggEngine::new();
        match engine.verify("(not (and p q))", "(or (not p) (not q))") {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("De Morgan #1 must verify, got Unproven: {reason}")
            }
        }
    }

    /// De Morgan #2: not(or(p,q)) ≡ and(not(p), not(q)).
    #[test]
    fn de_morgan_2_proves_equivalent() {
        let engine = EggEngine::new();
        match engine.verify("(not (or p q))", "(and (not p) (not q))") {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("De Morgan #2 must verify, got Unproven: {reason}")
            }
        }
    }

    /// Double-negation elimination: not(not(p)) ≡ p.
    #[test]
    fn double_negation_proves_equivalent() {
        let engine = EggEngine::new();
        match engine.verify("(not (not p))", "p") {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("double negation must verify, got Unproven: {reason}")
            }
        }
    }

    /// Non-equivalent boolean pair must stay Unproven: no rule in
    /// [`crate::rules::all_rules`] maps an `and` root onto an `or` root,
    /// so the two expressions remain in distinct e-classes.
    #[test]
    fn and_vs_or_stays_unproven() {
        let engine = EggEngine::new();
        match engine.verify("(and p q)", "(or p q)") {
            Verdict::Unproven { reason } => {
                assert_eq!(reason, "saturation exhausted without merge");
            }
            Verdict::Equivalent { .. } => panic!("and vs or must NOT be provable"),
        }
    }

    /// The algebraic identities added to the math lang: x+0 ≡ x, x*1 ≡ x
    /// (one-directional rules still prove equivalence — the e-graph keeps
    /// both the original node and its rewrite).
    #[test]
    fn algebraic_identities_prove_equivalent() {
        let engine = EggEngine::new();
        for (before, after) in [("(+ a 0)", "a"), ("(* a 1)", "a")] {
            match engine.verify(before, after) {
                Verdict::Equivalent { .. } => {}
                Verdict::Unproven { reason } => {
                    panic!("{before} vs {after} must verify, got Unproven: {reason}")
                }
            }
        }
    }

    // ---- Telemetry integration (§7.3–7.4) ----

    /// A green outcome flows through `verify_telemetry` with verdict_kind ==
    /// Green and REAL saturation metrics (iterations/nodes > 0); elapsed_ms
    /// is a non-negative wall-clock measurement.
    #[test]
    fn verify_telemetry_green_record_carries_metrics() {
        let engine = EggEngine::new();
        let record = engine.verify_telemetry(42, "(not (and p q))", "(or (not p) (not q))");
        assert_eq!(record.job_id(), 42);
        assert_eq!(
            record.verdict_kind(),
            crate::telemetry::VerdictKind::Green,
            "verdict_kind must match the verification outcome"
        );
        assert!(record.iterations() > 0, "iterations must be > 0");
        assert!(record.nodes() > 0, "nodes must be > 0");
        // `elapsed_ms: u128` is non-negative by type; bound it by the LAW
        // TIME_LIMIT (10s → 10_000ms) to prove it was actually measured.
        assert!(record.elapsed_ms() <= crate::limits::TIME_LIMIT.as_millis());

        // Composes with the existing sink — no duplicated sink logic.
        let mut sink = crate::telemetry::TelemetrySink::new();
        sink.record(record);
        assert_eq!(sink.green_yellow_ratio(), Some(1.0));
    }

    /// A yellow outcome maps to VerdictKind::Yellow; per the telemetry
    /// module contract, Unproven carries no metrics (0/0).
    #[test]
    fn verify_telemetry_yellow_record_maps_outcome() {
        let engine = EggEngine::new();
        let record = engine.verify_telemetry(7, "(and p q)", "(or p q)");
        assert_eq!(record.verdict_kind(), crate::telemetry::VerdictKind::Yellow);
        assert_eq!(record.iterations(), 0);
        assert_eq!(record.nodes(), 0);
        assert!(record.elapsed_ms() <= crate::limits::TIME_LIMIT.as_millis());

        let mut sink = crate::telemetry::TelemetrySink::new();
        sink.record(record);
        assert_eq!(sink.green_yellow_ratio(), Some(0.0));
    }

    // ---- SSA bridge verification (verify_ssa / verify_ssa_roots) ----

    /// Associativity: graph A computes (a+b)+c, graph B computes a+(b+c)
    /// over the same three params → Equivalent via ssa-assoc-add-flip.
    #[test]
    fn ssa_associativity_proves_equivalent() {
        // A: t = (a+b); out_a = t+c   → "(add (add p0 p1) p2)"
        let mut ga = Ssa::new();
        let (a0, b0, c0) = (ga.new_param(), ga.new_param(), ga.new_param());
        let t0 = ga.add(a0, b0);
        let _out_a = ga.add(t0, c0);
        // B: u = (b+c); out_b = a+u  → "(add p0 (add p1 p2))"
        let mut gb = Ssa::new();
        let (a1, b1, c1) = (gb.new_param(), gb.new_param(), gb.new_param());
        let u = gb.add(b1, c1);
        let out_b = gb.add(a1, u);

        let engine = EggEngine::new();
        match engine.verify_ssa(&ga, &gb, out_b) {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("SSA associativity must verify, got Unproven: {reason}")
            }
        }
    }

    /// Commutativity: add(a,b) vs add(b,a) → Equivalent via ssa-commute-add.
    #[test]
    fn ssa_commutativity_proves_equivalent() {
        let mut ga = Ssa::new();
        let (x, y) = (ga.new_param(), ga.new_param());
        let _out_a = ga.add(x, y);
        let mut gb = Ssa::new();
        let (y2, x2) = (gb.new_param(), gb.new_param());
        let out_b = gb.add(y2, x2);

        let engine = EggEngine::new();
        match engine.verify_ssa(&ga, &gb, out_b) {
            Verdict::Equivalent { .. } => {}
            Verdict::Unproven { reason } => {
                panic!("SSA commutativity must verify, got Unproven: {reason}")
            }
        }
    }

    /// Algebraic identity through the IR: A = add(a, const 0), B = bare
    /// param a → Equivalent via the `ssa-add-zero` rule ("(add ?a 0)" => ?a;
    /// constants emit as bare literals per ssa_bridge contract).
    #[test]
    fn ssa_add_zero_identity_proves_equivalent() {
        let mut ga = Ssa::new();
        let a = ga.new_param();
        let zero = ga.constant(0);
        let _out_a = ga.add(a, zero);
        let mut gb = Ssa::new();
        let b = gb.new_param();

        let engine = EggEngine::new();
        match engine.verify_ssa(&ga, &gb, b) {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("(a+0) vs a must verify, got Unproven: {reason}")
            }
        }
    }

    /// Genuinely different computations stay Unproven — DOCUMENTED
    /// LIMITATION: the §7.2 set has no `neg` cancellation rule, so B's extra
    /// `(add out (neg c))` tail can never dissolve; classes stay distinct.
    #[test]
    fn ssa_genuinely_different_stays_unproven() {
        let mut ga = Ssa::new();
        let (a0, b0) = (ga.new_param(), ga.new_param());
        let _out_a = ga.add(a0, b0);
        let mut gb = Ssa::new();
        let (a1, b1, c1) = (gb.new_param(), gb.new_param(), gb.new_param());
        let s = gb.add(a1, b1);
        let nz = gb.neg(c1); // no rule eliminates neg → no cancellation
        let out_b = gb.add(s, nz);

        let engine = EggEngine::new();
        match engine.verify_ssa(&ga, &gb, out_b) {
            Verdict::Unproven { reason } => {
                assert_eq!(reason, "saturation exhausted without merge");
            }
            Verdict::Equivalent { .. } => panic!("extra addend must NOT be provable"),
        }
    }

    /// Param naming collision safety: graphs with different param counts
    /// still convert (`p0..pN` keyed on per-graph declaration ordinals). Two roots
    /// that merely share an index but not structure stay Unproven instead of
    /// accidentally merging on names.
    #[test]
    fn ssa_param_count_mismatch_converts_without_cross_merge() {
        // A: one param + one def, root = add(x,x) → "(add p0 p0)".
        let mut ga = Ssa::new();
        let x = ga.new_param();
        let out_a = ga.add(x, x);
        // B: three params, root is the third param → "p2" (different atom!).
        let mut gb = Ssa::new();
        let _p = gb.new_param();
        let _q = gb.new_param();
        let r = gb.new_param();

        let engine = EggEngine::new();
        match engine.verify_ssa(&ga, &gb, r) {
            Verdict::Unproven { reason } => {
                assert_eq!(reason, "saturation exhausted without merge");
            }
            Verdict::Equivalent { .. } => panic!("p0 vs p2 must NOT merge across graphs"),
        }
        // Both conversions succeeded despite the count mismatch:
        assert_eq!(crate::ssa_bridge::to_rec_expr(&ga, x).to_string(), "p0");
        assert_eq!(crate::ssa_bridge::to_rec_expr(&gb, r).to_string(), "p2");
    }

    /// Identical graphs are trivially equivalent (same tree in one e-class),
    /// even if saturation finds nothing to rewrite.
    #[test]
    fn ssa_identical_graphs_are_equivalent() {
        let mut g = Ssa::new();
        let (a, b) = (g.new_param(), g.new_param());
        let m = g.mul(a, b);
        let out = g.mul(m, b);
        let engine = EggEngine::new();
        match engine.verify_ssa(&g, &g.clone(), out) {
            Verdict::Equivalent { nodes, .. } => assert!(nodes > 0),
            Verdict::Unproven { reason } => {
                panic!("identical graphs must verify, got Unproven: {reason}")
            }
        }
    }

    /// Degenerate input: a `before` with no defined operations is Unproven
    /// with an explicit reason, never a panic or empty-root surprise.
    #[test]
    fn ssa_empty_before_is_unproven_not_panic() {
        let mut ga = Ssa::new(); // params only, zero defs
        let _p = ga.new_param();
        let mut gb = Ssa::new();
        let q = gb.new_param();
        let engine = EggEngine::new();
        match engine.verify_ssa(&ga, &gb, q) {
            Verdict::Unproven { reason } => {
                assert!(reason.contains("defines no operations"));
            }
            Verdict::Equivalent { .. } => panic!("empty before cannot prove anything"),
        }
    }

    /// Explicit-roots form: same associativity pair proven when BOTH roots
    /// are named (here: the outputs are each graph's last def anyway).
    #[test]
    fn verify_ssa_roots_matches_inferred_roots() {
        let mut ga = Ssa::new();
        let (a0, b0, c0) = (ga.new_param(), ga.new_param(), ga.new_param());
        let l0 = ga.add(a0, b0);
        let out_a = ga.add(l0, c0);
        let mut gb = Ssa::new();
        let (a1, b1, c1) = (gb.new_param(), gb.new_param(), gb.new_param());
        let r0 = gb.add(b1, c1);
        let out_b = gb.add(a1, r0);

        let engine = EggEngine::new();
        match engine.verify_ssa_roots(&ga, out_a, &gb, out_b) {
            Verdict::Equivalent { .. } => {}
            Verdict::Unproven { reason } => {
                panic!("explicit-root form must verify, got Unproven: {reason}")
            }
        }
    }

    /// Telemetry composition for the SSA path mirrors verify_telemetry:
    /// green carries real metrics, yellow carries none; both land in the
    /// shared sink untouched.
    #[test]
    fn verify_ssa_telemetry_records_green_and_yellow() {
        let mut ga = Ssa::new();
        let (a0, b0, c0) = (ga.new_param(), ga.new_param(), ga.new_param());
        let l0 = ga.add(a0, b0);
        let _out_a = ga.add(l0, c0);
        let mut gb = Ssa::new();
        let (a1, b1, c1) = (gb.new_param(), gb.new_param(), gb.new_param());
        let r0 = gb.add(b1, c1);
        let out_b = gb.add(a1, r0);

        // Genuinely different pair (see ssa_genuinely_different_stays_unproven).
        let mut gc = Ssa::new();
        let (x, y) = (gc.new_param(), gc.new_param());
        let _out_c = gc.mul(x, y);

        let engine = EggEngine::new();
        let green = engine.verify_ssa_telemetry(1, &ga, &gb, out_b);
        assert_eq!(green.job_id(), 1);
        assert_eq!(green.verdict_kind(), crate::telemetry::VerdictKind::Green);
        assert!(green.iterations() > 0);
        assert!(green.nodes() > 0);
        assert!(green.elapsed_ms() <= crate::limits::TIME_LIMIT.as_millis());

        let yellow = engine.verify_ssa_telemetry(2, &ga, &gc, Value(
            (gc.len() - 1) as u32,
        ));
        assert_eq!(yellow.verdict_kind(), crate::telemetry::VerdictKind::Yellow);
        assert_eq!(yellow.iterations(), 0);
        assert_eq!(yellow.nodes(), 0);

        let mut sink = crate::telemetry::TelemetrySink::new();
        sink.record(green);
        sink.record(yellow);
        assert_eq!(sink.green_yellow_ratio(), Some(0.5));
    }
}
