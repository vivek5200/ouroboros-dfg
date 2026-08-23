#![cfg(feature = "pyo3-ext")]
//! PyO3 bindings for the Ouroboros v7.1 DFG verifier.
//!
//! Exposes [`crate::engine::EggEngine`] to Python as the `ouroboros_dfg`
//! extension module. This module is a thin adapter ONLY: every verification
//! still runs through the synchronous, LAW-bounded engine
//! ([`crate::limits::RunnerLimits::law_mandated`]), composed by ownership —
//! no rewrite system, runner, or limit is duplicated here.
//!
//! Build (never on by default, so `cargo test` stays pure-Rust):
//!
//! ```bash
//! python3 -m maturin build --release --features pyo3-ext
//! python3 -m pip install --user --break-system-packages target/wheels/*.whl
//! python3 -c 'import ouroboros_dfg; e = ouroboros_dfg.PyEggEngine(); \
//!              print(e.verify("(+ a b)", "(+ b a)")); print(e.law_limits())'
//! ```
//!
//! Source-level surface: [`prove_refactor`] takes two REAL Python source
//! strings and returns the verdict dict of
//! [`crate::e2e_proof::prove_refactor`] — no Rust or s-expression knowledge
//! needed on the Python side:
//!
//! ```python
//! ouroboros_dfg.prove_refactor("r = (a + b) + c", "r = a + (b + c)")
//! # {'verdict': 'equivalent', 'iterations': …, 'nodes': …,
//! #  'lowered_before_len': 5, 'lowered_after_len': 5}
//! ```

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyString};

use crate::engine::{EggEngine, Verdict};

/// Helper: lift a Rust `&str` into an owned Python object.
fn py_str(py: Python<'_>, s: &str) -> Py<PyAny> {
    PyString::new(py, s).into_any().unbind()
}

/// Helper: lift a `usize` counter into an owned Python int.
fn py_usize(py: Python<'_>, n: usize) -> PyResult<Py<PyAny>> {
    Ok(n.into_pyobject(py)?.unbind().into())
}

/// Python-facing equivalence checker.
///
/// Wraps one [`EggEngine`] pinned to `RunnerLimits::law_mandated()`;
/// construction cannot deviate from the egraph-limits LAW because the
/// inner engine offers no other constructor path.
///
/// ```python
/// engine = ouroboros_dfg.PyEggEngine()
/// engine.verify("(+ a b)", "(+ b a)")   # {'verdict': 'equivalent', ...}
/// ```
#[pyclass(module = "ouroboros_dfg")]
struct PyEggEngine {
    inner: EggEngine,
}

#[pymethods]
impl PyEggEngine {
    /// Create an engine pinned to the LAW-mandated resource envelope.
    #[new]
    fn new() -> Self {
        Self {
            inner: EggEngine::new(),
        }
    }

    /// Verify whether `before` and `after` are semantically equivalent
    /// under the AC-add rewrite system, within the LAW limits.
    ///
    /// Returns a dict:
    /// - equivalent: `{"verdict": "equivalent", "iterations": int, "nodes": int}`
    /// - unproven:   `{"verdict": "unproven", "reason": str}`
    fn verify(&self, before: &str, after: &str) -> PyResult<HashMap<String, Py<PyAny>>> {
        // Pure-Rust computation happens outside the GIL region below; the
        // engine itself takes no Python objects.
        let verdict = self.inner.verify(before, after);

        Python::attach(|py| {
            let mut result: HashMap<String, Py<PyAny>> = HashMap::new();
            match verdict {
                Verdict::Equivalent { iterations, nodes } => {
                    result.insert("verdict".to_string(), py_str(py, "equivalent"));
                    result.insert("iterations".to_string(), py_usize(py, iterations)?);
                    result.insert("nodes".to_string(), py_usize(py, nodes)?);
                }
                Verdict::Unproven { reason } => {
                    result.insert("verdict".to_string(), py_str(py, "unproven"));
                    result.insert("reason".to_string(), py_str(py, &reason));
                }
            }
            Ok(result)
        })
    }

    /// The five pinned egraph-limits LAW bounds, one human-readable line each.
    ///
    /// Mirrors `crate::limits`: IterationLimit 5000, TimeLimit 10s,
    /// NodeLimit 1_000_000, BackoffScheduler{match_limit 5000, ban_length 3}.
    fn law_limits(&self) -> Vec<String> {
        let limits = self.inner.limits();
        vec![
            format!(
                "iteration_limit = {} (egg Runner::with_iter_limit)",
                limits.iteration_limit
            ),
            format!(
                "time_limit = {}s (egg Runner::with_time_limit)",
                limits.time_limit.as_secs()
            ),
            format!(
                "node_limit = {} (egg Runner::with_node_limit)",
                limits.node_limit
            ),
            format!(
                "backoff_match_limit = {} (BackoffScheduler::with_initial_match_limit)",
                limits.backoff_match_limit
            ),
            format!(
                "backoff_ban_length = {} (BackoffScheduler::with_ban_length)",
                limits.backoff_ban_length
            ),
        ]
    }
}

/// Prove (or fail to prove) that `after_src` is a semantics-preserving
/// refactor of `before_src`, both given as REAL Python source text — the
/// source-level counterpart of [`PyEggEngine::verify`] (which speaks
/// s-expression terms instead).
///
/// Each source is lowered by the Module 5 tree-sitter frontend and verified
/// through [`crate::e2e_proof::prove_refactor`] under the LAW-bounded
/// engine. The per-side OUTPUT VALUE is resolved by that function's
/// documented chain: empty name hints fall through to
/// [`Lowered::last`](crate::frontend_python::Lowered::last) — the final bare
/// expression statement — then to `crate::ssa_bridge::last_defined`, i.e.
/// each source's LAST ASSIGNED variable. Plain `name = expr` modules
/// therefore need no out-value hints from the Python caller.
///
/// Returns a dict:
/// - equivalent: `{"verdict": "equivalent", "iterations": int, "nodes": int,
///   "lowered_before_len": int, "lowered_after_len": int}`
/// - unproven:   same keys with `"iterations": None` and `"nodes": None`
///   (telemetry convention: yellow carries no metrics)
///
/// Raises `ValueError` when a source is outside the supported Python subset
/// or no output value can be resolved for one side.
#[pyfunction]
fn prove_refactor(before_src: &str, after_src: &str) -> PyResult<HashMap<String, Py<PyAny>>> {
    // Pure-Rust computation happens outside the GIL region below; the proof
    // harness takes no Python objects. Empty name hints delegate output
    // resolution entirely to the e2e_proof chain documented above.
    let report = crate::e2e_proof::prove_refactor(before_src, after_src, "", "")
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // ProofReport pins only the BEFORE graph size; the after-side size is
    // recomputed by re-running the pure, deterministic frontend on the
    // already-accepted source (cannot fail here: prove_refactor just
    // lowered it successfully).
    let lowered_after_len = crate::frontend_python::lower_module(after_src)
        .map(|l| l.ssa.len())
        .unwrap_or(0);

    Python::attach(|py| {
        let mut result: HashMap<String, Py<PyAny>> = HashMap::new();
        match report.verdict {
            Verdict::Equivalent { iterations, nodes } => {
                result.insert("verdict".to_string(), py_str(py, "equivalent"));
                result.insert("iterations".to_string(), py_usize(py, iterations)?);
                result.insert("nodes".to_string(), py_usize(py, nodes)?);
            }
            Verdict::Unproven { .. } => {
                result.insert("verdict".to_string(), py_str(py, "unproven"));
                result.insert("iterations".to_string(), py.None());
                result.insert("nodes".to_string(), py.None());
            }
        }
        result.insert(
            "lowered_before_len".to_string(),
            py_usize(py, report.lowered_before_len)?,
        );
        result.insert(
            "lowered_after_len".to_string(),
            py_usize(py, lowered_after_len)?,
        );
        Ok(result)
    })
}

/// The `ouroboros_dfg` Python extension module.
#[pymodule]
fn ouroboros_dfg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEggEngine>()?;
    m.add_function(wrap_pyfunction!(prove_refactor, m)?)?;
    Ok(())
}

// NOTE on testing: a `#[cfg(test)]` unit test for `prove_refactor` is NOT
// possible here — `pyo3-ext` implies `pyo3/extension-module`, under which a
// standalone test binary fails to LINK (undefined `PyBool_Type`,
// `Py_IsInitialized`, …: libpython is intentionally not linked outside a
// Python process; verified empirically with `cargo test --features
// pyo3-ext`). The dict contract is therefore covered by the real-Python
// smoke after every maturin rebuild (equivalent + unproven + ValueError).
