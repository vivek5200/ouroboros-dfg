//! End-to-end proof harness — the first FULL-CHAIN proof on real Python
//! source text: two refactoring variants enter as strings, a mathematical
//! equivalence verdict exits through every Module 5 stage:
//!
//! ```text
//! python source ─frontend_python::lower_module→ SSA graphs (one per side)
//!               ─ssa_bridge::to_rec_expr→      egg RecExprs
//!               ─engine::verify_ssa_roots→     Verdict (+ saturation metrics)
//! ```
//!
//! Unlike [`crate::engine`] tests, which hand-build `Ssa` values through the
//! builder API, EVERYTHING here starts from raw source text lowered by the
//! tree-sitter frontend ([`crate::frontend_python`]) — assignments, integer
//! arithmetic (`+ - *`) and parentheses, i.e. exactly the subset the
//! frontend documents as supported.
//!
//! SIGNATURE NOTE (documented deviation from the task sketch): the sketch
//! wrote `out_before: Value, out_after: Value`, but a source-level API
//! caller CANNOT know graph-local [`Value`] ids before lowering — they are
//! minted per-graph during [`frontend_python::lower_module`]. The out
//! parameters are therefore the OUTPUT VARIABLE NAMES (`&str`). Each name is
//! resolved against its own lowered graph with this documented chain:
//!
//! 1. [`Lowered::value_of`](`crate::frontend_python::Lowered::value_of`) —
//!    the primary path: the final binding of the named variable;
//! 2. [`Lowered.last`](crate::frontend_python::Lowered::last) — the value of
//!    the final bare expression statement (used when no name is given, i.e.
//!    the empty string);
//! 3. [`crate::ssa_bridge::last_defined`] — the sanctioned FALLBACK: the
//!    last assigned variable's value (highest defined id), for sources that
//!    neither bind the name nor end in an expression statement.
//!
//! THE CONTROL CASE (documented per task): the non-equivalent pair is
//! `r = a * 2` vs `r = a + 2` → [`Verdict::Unproven`]. This pair is
//! genuinely different under the CURRENT rule set AND in ordinary integer
//! arithmetic: `SymbolLang` operators are opaque symbols, so the `(mul p0 2)`
//! and `(add p0 2)` roots can never merge (no rule maps `mul`↔`add`; the
//! identities need literals `0`/`1`). The alternative sketched in the task,
//! `r = a + b` vs `r = a - b`, would ALSO stay Unproven (binary `-` lowers
//! to `add(a, neg(b))` and there is no `neg`-cancellation rule — see the
//! documented limitation on `ssa_genuinely_different_stays_unproven` in
//! [`crate::engine`] tests), but the mul-vs-add control does not depend on
//! absent `neg` rules, so it isolates "different computation" cleanly.
//!
//! RULE COVERAGE NOTE: proving `r = a * b ≡ r = b * a` required adding
//! `commute-mul` / `ssa-commute-mul` to [`crate::rules`] (child-swap is
//! involutive — same one-direction argument as `commute-add`). Without it
//! the two muls sit in distinct e-classes forever and the `commute` case
//! below could only ever return Unproven.

use std::fmt;

use crate::engine::{EggEngine, Verdict};
use crate::frontend_python::{self, Lowered, LoweringError};
use crate::ssa::Value;

/// Machine-readable result of one end-to-end refactor proof: what went in
/// (both sources verbatim), how big the lowered `before` graph was, and the
/// verdict with its saturation metrics.
///
/// Metric convention mirrors [`crate::telemetry`] (yellow carries no
/// metrics): an [`Verdict::Unproven`] outcome reports `iterations == 0` and
/// `nodes == 0`; an [`Verdict::Equivalent`] outcome mirrors the runner's
/// real numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofReport {
    /// The `before` source text, verbatim as passed in.
    pub before_src: String,
    /// The `after` source text, verbatim as passed in.
    pub after_src: String,
    /// Number of SSA values (params + defs) minted while lowering
    /// `before_src` — the size of the verified graph.
    pub lowered_before_len: usize,
    /// The mathematical verdict from [`EggEngine::verify_ssa_roots`].
    pub verdict: Verdict,
    /// Rewriting iterations performed (0 when Unproven).
    pub iterations: usize,
    /// E-graph enodes at the final recorded iteration (0 when Unproven).
    pub nodes: usize,
}

/// Failure modes of [`prove_refactor`]: the frontend rejected a source, or
/// no output value could be resolved for one side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    /// One of the two sources is outside the supported Python subset
    /// (unsupported construct, syntax error, or out-of-range literal).
    Lowering(LoweringError),
    /// The output-value resolution chain (module docs) found nothing.
    MissingOutput {
        /// Which side failed: `"before"` or `"after"`.
        side: &'static str,
        /// Human-readable explanation.
        detail: String,
    },
}

impl fmt::Display for ProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProofError::Lowering(e) => write!(f, "{e}"),
            ProofError::MissingOutput { side, detail } => {
                write!(f, "no output value for `{side}` source: {detail}")
            }
        }
    }
}

impl std::error::Error for ProofError {}

impl From<LoweringError> for ProofError {
    fn from(e: LoweringError) -> Self {
        ProofError::Lowering(e)
    }
}

/// Resolve the output [`Value`] of one lowered source per the module-docs
/// chain: named binding → final expression statement → last defined value.
fn extract_out(lowered: &Lowered, name: &str, side: &'static str) -> Result<Value, ProofError> {
    if !name.is_empty() {
        if let Some(v) = lowered.value_of(name) {
            return Ok(v);
        }
    }
    if let Some(last) = lowered.last {
        return Ok(last);
    }
    if let Some(last_def) = crate::ssa_bridge::last_defined(&lowered.ssa) {
        return Ok(last_def);
    }
    Err(ProofError::MissingOutput {
        side,
        detail: format!(
            "variable `{name}` is unbound, no trailing expression statement, \
             and the graph defines no operations"
        ),
    })
}

/// Prove (or fail to prove) that `after_src` is a semantics-preserving
/// refactor of `before_src`, BOTH given as real Python source text.
///
/// Both sources are lowered independently via
/// [`frontend_python::lower_module`]; their output values are resolved with
/// [`extract_out`] using `out_before` / `out_after` as variable names
/// (empty string ⇒ fall through to `last`/`last_defined` — see the module
/// docs for the full chain and the rationale for name-based parameters).
/// The two graphs then go through [`EggEngine::verify_ssa_roots`] under the
/// LAW-bounded limits and the [`crate::rules::ssa_all_rules`] system.
///
/// Returns [`ProofReport`] on a completed run — note that a run completing
/// with distinct e-classes is STILL `Ok`: the verdict itself carries the
/// Unproven outcome. Only lowering/resolution failures are `Err`.
pub fn prove_refactor(
    before_src: &str,
    after_src: &str,
    out_before: &str,
    out_after: &str,
) -> Result<ProofReport, ProofError> {
    let before = frontend_python::lower_module(before_src)?;
    let after = frontend_python::lower_module(after_src)?;

    let root_before = extract_out(&before, out_before, "before")?;
    let root_after = extract_out(&after, out_after, "after")?;

    let verdict = EggEngine::new()
        .verify_ssa_roots(&before.ssa, root_before, &after.ssa, root_after);

    let (iterations, nodes) = match &verdict {
        Verdict::Equivalent { iterations, nodes } => (*iterations, *nodes),
        Verdict::Unproven { .. } => (0, 0),
    };

    Ok(ProofReport {
        before_src: before_src.to_string(),
        after_src: after_src.to_string(),
        lowered_before_len: before.ssa.len(),
        verdict,
        iterations,
        nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CASE 1 — associativity through parens: `r = (a + b) + c` ≡
    /// `r = a + (b + c)`. Both sides lower to nested adds over the SAME
    /// three params (`(add (add p0 p1) p2)` vs `(add p0 (add p1 p2))`);
    /// `ssa-assoc-add-flip` collapses them into one e-class → Equivalent
    /// with real saturation metrics.
    #[test]
    fn e2e_assoc_paren_regrouping_is_equivalent() {
        let before = "r = (a + b) + c";
        let after = "r = a + (b + c)";
        let report = prove_refactor(before, after, "r", "r").unwrap();

        match report.verdict {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1, "saturation must have run");
                assert!(nodes > 0, "egraph must hold nodes");
                // Report mirrors the verdict metrics verbatim.
                assert_eq!(report.iterations, iterations);
                assert_eq!(report.nodes, nodes);
            }
            Verdict::Unproven { reason } => {
                panic!("assoc regrouping must prove, got Unproven: {reason}")
            }
        }
        // Report round-trips the inputs and the real lowered size:
        // %0=a %1=b %2=c (params) + %3=(a+b) + %4=(%3+c) = 5 values.
        assert_eq!(report.before_src, before);
        assert_eq!(report.after_src, after);
        assert_eq!(report.lowered_before_len, 5);
    }

    /// CASE 2 — commutativity: `r = a * b` ≡ `r = b * a`. Requires the
    /// `ssa-commute-mul` rule added to [`crate::rules::ssa_math_rules`]
    /// (see module docs); `(mul p0 p1)` swaps to `(mul p1 p0)` → Equivalent.
    #[test]
    fn e2e_commuted_multiplication_is_equivalent() {
        let report = prove_refactor("r = a * b", "r = b * a", "r", "r").unwrap();
        match report.verdict {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("mul commutation must prove, got Unproven: {reason}")
            }
        }
        // Both sides lower to exactly two params + one def.
        assert_eq!(report.lowered_before_len, 3);
    }

    /// CASE 3 — algebraic identity: `r = x + 0` ≡ `r = x`. `x` is an input
    /// parameter (frontend NAME POLICY), constants emit as bare literals, so
    /// `ssa-add-zero` ((add ?a 0) => ?a) reduces the left side onto the bare
    /// param `p0` that IS the right side's root → Equivalent.
    #[test]
    fn e2e_add_zero_identity_is_equivalent() {
        let report = prove_refactor("r = x + 0", "r = x", "r", "r").unwrap();
        match report.verdict {
            Verdict::Equivalent { iterations, nodes } => {
                assert!(iterations >= 1);
                assert!(nodes > 0);
            }
            Verdict::Unproven { reason } => {
                panic!("x + 0 ≡ x must prove, got Unproven: {reason}")
            }
        }
    }

    /// CASE 4 — non-equivalent CONTROL (documented in module docs):
    /// `r = a * 2` vs `r = a + 2`. Opaque `mul`/`add` roots live in disjoint
    /// families, no rule crosses them, and the programs genuinely differ →
    /// Unproven with the exact classification reason and ZEROED metrics
    /// (telemetry convention: yellow carries no metrics).
    #[test]
    fn e2e_mul_vs_add_control_stays_unproven() {
        let report = prove_refactor("r = a * 2", "r = a + 2", "r", "r").unwrap();
        match &report.verdict {
            Verdict::Unproven { reason } => {
                assert_eq!(reason, "saturation exhausted without merge");
            }
            Verdict::Equivalent { .. } => {
                panic!("a*2 is NOT a refactor of a+2; must NOT prove")
            }
        }
        assert_eq!(report.iterations, 0);
        assert_eq!(report.nodes, 0);
        assert_eq!(report.lowered_before_len, 3); // p0, const 2, the def
    }

    /// The documented extraction FALLBACK ORDER: with empty name hints the
    /// resolver uses each side's final bare expression statement
    /// ([`Lowered::last`]) — here the assoc pair still proves Equivalent.
    #[test]
    fn e2e_bare_expression_statements_resolve_via_last() {
        let report = prove_refactor("(a + b) + c", "a + (b + c)", "", "").unwrap();
        assert!(matches!(report.verdict, Verdict::Equivalent { .. }));

        // Second fallback rung: assignments only + empty names resolve to
        // the LAST ASSIGNED variable (last_defined) — the sanctioned
        // fallback from the task description.
        let assigned = prove_refactor("t = a * b", "t = b * a", "", "").unwrap();
        assert!(matches!(assigned.verdict, Verdict::Equivalent { .. }));
    }

    /// Frontend rejection propagates as [`ProofError::Lowering`]: floats are
    /// outside the supported subset, so NO verdict is produced at all.
    #[test]
    fn e2e_lowering_error_is_reported_not_proven() {
        let err = prove_refactor("r = 1.5 + a", "r = a + 1", "r", "r").unwrap_err();
        match err {
            ProofError::Lowering(e) => {
                assert!(e.message.contains("unsupported"), "got: {e}");
            }
            other => panic!("expected Lowering error, got {other:?}"),
        }
    }

    /// A source with no resolvable output (empty module) yields
    /// [`ProofError::MissingOutput`] naming the failing side — never a panic.
    #[test]
    fn e2e_missing_output_names_the_side() {
        let err = prove_refactor("", "r = 1", "", "r").unwrap_err();
        match &err {
            ProofError::MissingOutput { side, detail } => {
                assert_eq!(*side, "before");
                assert!(detail.contains("unbound"), "{detail}");
            }
            other => panic!("expected MissingOutput, got {other:?}"),
        }
        // Display renders a human-readable message either way.
        assert!(err.to_string().contains("before"));
    }

    /// Full-chain sanity: an equivalent pair evaluated CONCRETELY agrees —
    /// the SSA graphs behind a green verdict compute identical values, so
    /// the proof is about real data flow, not just symbol shuffling.
    #[test]
    fn e2e_green_verdict_graphs_agree_concretely() {
        let report = prove_refactor("r = (a + b) + c", "r = a + (b + c)", "r", "r").unwrap();
        assert!(matches!(report.verdict, Verdict::Equivalent { .. }));

        let lb = frontend_python::lower_module("r = (a + b) + c").unwrap();
        let la = frontend_python::lower_module("r = a + (b + c)").unwrap();
        let rb = lb.value_of("r").unwrap();
        let ra = la.value_of("r").unwrap();
        assert_eq!(lb.ssa.evaluate(rb, &[2, 3, 4]).unwrap(), 9);
        assert_eq!(la.ssa.evaluate(ra, &[2, 3, 4]).unwrap(), 9);
    }
}
