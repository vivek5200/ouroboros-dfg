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

use std::collections::HashMap;

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

/// The `ouroboros_dfg` Python extension module.
#[pymodule]
fn ouroboros_dfg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEggEngine>()?;
    Ok(())
}
