//! Module 5: DFG Verification — Ouroboros v7.1 semantic verifier.
//!
//! Extracts SSA-form Data Flow Graphs from Python/C++ sources and verifies
//! semantic equivalence via egg equality saturation.
//!
//! LAW (egraph-limits, confidence 0.99): every `egg::Runner` in this crate
//! MUST be constructed through [`limits`] with the hardcoded bounds
//! (IterationLimit 5000, TimeLimit 10s, NodeLimit 1_000_000,
//! BackoffScheduler{match_limit 5000, ban_length 3}), and equality
//! saturation must run strictly as an asynchronous background CPU task —
//! never blocking GPU inference or training loops.

/// Optional PyO3 bindings (`ouroboros_dfg` Python extension module).
/// Off by default — enable with `--features pyo3-ext`; see module docs.
#[cfg(feature = "pyo3-ext")]
pub mod pyo3_ext;

pub mod async_engine;
/// Module 5 Python frontend: tree-sitter lowering of a constrained but real
/// subset (integer arithmetic + assignment) into [`crate::ssa`].
pub mod frontend_python;
pub mod engine;
pub mod limits;
pub mod rules;
pub mod ssa;
/// SSA → egg lowering ([`ssa_bridge::to_rec_expr`]): the ONLY egg glue for
/// the IR, keeping [`crate::ssa`] stdlib-pure; consumed by [`crate::engine`].
pub mod ssa_bridge;
pub mod telemetry;
