//! SSA → egg bridge: lowers [`crate::ssa::Ssa`] data flow graphs into
//! [`egg::RecExpr<SymbolLang>`] s-expressions so SSA-level equivalence can
//! flow through the existing LAW-bounded engine ([`crate::engine`]).
//!
//! DESIGN DECISION (documented per task instructions): [`crate::ssa`] stays
//! stdlib-pure — its module docs promise "no egg/tokio dependency" so the IR
//! remains buildable in any frontend context. ALL egg glue therefore lives
//! HERE (the "small ssa_bridge module" option), and [`crate::engine`] is the
//! only consumer, via [`to_rec_expr`] + [`last_defined`].
//!
//! EMISSION CONTRACT:
//! - op spellings: `add` / `mul` / `neg`, each def renders as
//!   `(op arg1 arg2 ...)`;
//! - parameters render as `p<N>` where `N` is the parameter's **declaration
//!   ordinal** ([`Ssa::param_ordinal`], i.e. "i-th declared input"), NOT its
//!   raw Value id. RATIONALE (found by the Module 5 end-to-end Python proof,
//!   [`crate::e2e_proof`]): frontends mint defs and params interleaved in
//!   evaluation order, so `r = (a + b) + c` allocates c's Value id AFTER the
//!   intermediate add (`p3`) while `r = a + (b + c)` allocates it before
//!   (`p2`). Id-keyed names would leave the same input spelled differently
//!   on each side and NO refactoring could ever cross-merge. Ordinals are
//!   dense, graph-local, and equal on both sides whenever the programs read
//!   the same number of inputs in the same order; graphs with different
//!   arities still never cross-contaminate names;
//! - constants render as bare integer literals (`0`, `7`, `-3`) — this is
//!   what lets the algebraic identities in [`crate::rules::ssa_math_rules`]
//!   (`(add ?a 0)` => `?a`) match emitted graphs;
//! - memoization over Value ids caches each rendered subterm exactly once
//!   per unique id, so DAG-shaped IR renders in time linear in the number
//!   of *edges* rather than re-walking shared subtrees;
//! - dangling operands (`Value` ≥ `ssa.len()`, i.e. inputs that would fail
//!   [`crate::ssa::Ssa::validate`]) render as a unique inert atom `undef<N>`
//!   instead of panicking, so a malformed graph converts to something that
//!   simply cannot merge; cyclic IR (unvalidatable by construction today)
//!   cuts cycles at the first revisit with a `v<N>cycle` atom, guaranteeing
//!   termination.
//!
//! EGG API SOURCES (verified against local crate source, not guessed):
//! - `~/.cargo/registry/src/*/egg-0.9.5/src/language.rs:545`
//!     - `impl<L: FromOp> FromStr for RecExpr<L>`: recursive descent over the
//!       sexp calling `L::from_op` per node (lines 551–572).
//! - `~/.cargo/registry/src/*/egg-0.9.5/src/language.rs:847–857`
//!     - `FromOp for SymbolLang` returns `Ok` unconditionally (`Infallible`),
//!       so parsing our output can only fail on *malformed structure*, which
//!       the renderer below cannot produce (balanced parens, bare-token
//!       atoms). This justifies the single `expect` in [`to_rec_expr`].
//! - `~/.cargo/registry/src/*/egg-0.9.5/src/language.rs:457–468`
//!     - `Display for RecExpr` renders via `to_sexp()`; tests use it to pin
//!       the exact emitted form.

use std::collections::{HashMap, HashSet};

use egg::{RecExpr, SymbolLang};

use crate::ssa::{Op, Ssa, Value};

/// Lower `result` (a value of `ssa`) into an s-expression e-graph term.
///
/// See the module docs for the exact emission contract. Panics only on an
/// internal renderer bug (structurally malformed output), which the SymbolLang
/// parser source rules out for well-formed sexps (language.rs:545–572).
pub fn to_rec_expr(ssa: &Ssa, result: Value) -> RecExpr<SymbolLang> {
    let text = render(
        ssa,
        result,
        &mut HashMap::new(),
        &mut HashSet::new(),
    );
    text.parse()
        .expect("renderer emits balanced, atom-only sexps; parse is infallible")
}

/// The natural root of a builder-order graph: its highest defined value.
///
/// `None` when the graph defines no operations at all (params only / empty).
/// Used by [`crate::engine::EggEngine::verify_ssa`] so callers only need to
/// name the `after` root explicitly.
pub fn last_defined(ssa: &Ssa) -> Option<Value> {
    (0..ssa.len() as u32)
        .rev()
        .find(|i| ssa.op_of(Value(*i)).is_some())
        .map(Value)
}

/// Memoized renderer. `memo` maps Value id → rendered text; `open` holds ids
/// on the current recursion stack (cycle guard).
fn render(
    ssa: &Ssa,
    v: Value,
    memo: &mut HashMap<u32, String>,
    open: &mut HashSet<u32>,
) -> String {
    if let Some(hit) = memo.get(&v.0) {
        return hit.clone();
    }
    if !open.insert(v.0) {
        // Cycle cut: only reachable on IR that fails validate(); the atom is
        // unique per id so two broken graphs cannot spuriously merge here.
        return format!("v{}cycle", v.0);
    }
    let text = if ssa.is_param(v) {
        // is_param is bounds-checked (ssa.rs), so params are distinguished
        // from dangling ids before the match below. Names key on the
        // declaration ordinal (see EMISSION CONTRACT): `is_param` ⇒ the
        // value IS in `params`, so the ordinal lookup cannot miss — the
        // fallback to the raw id only guards an internal-invariant break.
        format!("p{}", ssa.param_ordinal(v).unwrap_or(v.0 as usize))
    } else {
        match ssa.op_of(v) {
            Some(Op::Const(c)) => c.to_string(),
            Some(Op::Add(a, b)) => {
                format!("(add {} {})", render(ssa, *a, memo, open), render(ssa, *b, memo, open))
            }
            Some(Op::Mul(a, b)) => {
                format!("(mul {} {})", render(ssa, *a, memo, open), render(ssa, *b, memo, open))
            }
            Some(Op::Neg(a)) => format!("(neg {})", render(ssa, *a, memo, open)),
            // Out-of-range Value (defs.len() <= v.0): inert unique atom.
            None => format!("undef{}", v.0),
        }
    };
    open.remove(&v.0);
    memo.insert(v.0, text.clone());
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// %0=param %1=param %2=param; t=%0+%1; out=t+%2 →
    /// "(add (add p0 p1) p2)" — nested defs become nested s-exprs.
    #[test]
    fn associativity_shape_emits_nested_sexprs() {
        let mut g = Ssa::new();
        let (a, b, c) = (g.new_param(), g.new_param(), g.new_param());
        let t = g.add(a, b);
        let out = g.add(t, c);
        assert_eq!(to_rec_expr(&g, out).to_string(), "(add (add p0 p1) p2)");
    }

    /// Constants emit as bare literals (so `(add ?a 0)` rules match), neg
    /// emits unary, mul emits binary.
    #[test]
    fn const_neg_mul_spellings() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let zero = g.constant(0);
        let s = g.add(a, zero);
        let neg3 = g.constant(-3);
        let m = g.mul(s, neg3);
        let n = g.neg(m);
        assert_eq!(
            to_rec_expr(&g, n).to_string(),
            "(neg (mul (add p0 0) -3))"
        );
    }

    /// A param-only root renders as its own `p<N>` atom.
    #[test]
    fn param_root_is_bare_atom() {
        let mut g = Ssa::new();
        let _a = g.new_param();
        let b = g.new_param();
        assert_eq!(to_rec_expr(&g, b).to_string(), "p1");
    }

    /// Memoization dedup: every occurrence of a shared subterm renders to the
    /// IDENTICAL text (one cache entry per Value id), and a diamond DAG keeps
    /// that consistency across three use sites.
    #[test]
    fn repeated_subterms_render_identically() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let seven = g.constant(7);
        let b = g.add(a, seven); // (add p0 7)
        let c = g.add(b, b); // uses b twice
        let out = g.add(c, b); // third use of b
        assert_eq!(
            to_rec_expr(&g, out).to_string(),
            "(add (add (add p0 7) (add p0 7)) (add p0 7))"
        );
    }

    /// Graphs with different param counts convert independently: names are
    /// minted per-graph from Value ids, so neither side references the other.
    #[test]
    fn different_param_counts_convert_without_collision() {
        let mut one = Ssa::new();
        let x = one.new_param();
        assert_eq!(to_rec_expr(&one, x).to_string(), "p0");

        let mut three = Ssa::new();
        let _p = three.new_param();
        let _q = three.new_param();
        let r = three.new_param();
        let s = three.add(r, r);
        assert_eq!(to_rec_expr(&three, s).to_string(), "(add p2 p2)");
    }

    /// REGRESSION (found by the Module 5 end-to-end Python proof): a param
    /// minted AFTER defs must still render with its dense DECLARATION
    /// ORDINAL. Here %0=a %1=b %2=(a+b) def %3=c ⇒ c is `p2`, NOT `p3`.
    #[test]
    fn params_interleaved_with_defs_name_by_ordinal() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let b = g.new_param();
        let t = g.add(a, b);
        let c = g.new_param(); // Value id 3, but third declared input
        let out = g.add(t, c);
        assert_eq!(to_rec_expr(&g, out).to_string(), "(add (add p0 p1) p2)");
    }

    /// Cross-graph name stability — the exact shape pair behind the
    /// associativity source refactor: both programs read the SAME three
    /// inputs in the same order, only the evaluation tree differs, so every
    /// input keeps its name on both sides and the rewrite can merge them.
    #[test]
    fn same_inputs_across_evaluation_shapes_render_matching_names() {
        // Left: `(a + b) + c` — c's Value id lands AFTER the intermediate def.
        let mut left = Ssa::new();
        let (a0, b0) = (left.new_param(), left.new_param());
        let t = left.add(a0, b0);
        let c0 = left.new_param();
        let lout = left.add(t, c0);
        // Right: `a + (b + c)` — all params first, then defs.
        let mut right = Ssa::new();
        let (a1, b1, c1) = (right.new_param(), right.new_param(), right.new_param());
        let u = right.add(b1, c1);
        let rout = right.add(a1, u);

        assert_eq!(to_rec_expr(&left, lout).to_string(), "(add (add p0 p1) p2)");
        assert_eq!(to_rec_expr(&right, rout).to_string(), "(add p0 (add p1 p2))");
    }

    /// Dangling operand: converts to an inert `undef<N>` atom (no panic),
    /// distinct from every legal param name.
    #[test]
    fn dangling_value_renders_inert_atom() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let s = g.add(a, Value(99));
        assert_eq!(to_rec_expr(&g, s).to_string(), "(add p0 undef99)");
        // Empty graph, arbitrary id.
        assert_eq!(to_rec_expr(&Ssa::new(), Value(4)).to_string(), "undef4");
    }

    /// Cycle cut terminates on IR that validate() rejects. A genuine cycle is
    /// constructible through the public API via a forward ref: %0 = add(%1,%1)
    /// (dangling at construction), then %1 = add(%0,%0) closes the loop.
    #[test]
    fn cycle_terminates_with_cut_atom() {
        let mut g = Ssa::new();
        let forward = g.add(Value(1), Value(1)); // %0 → %1 (not yet defined)
        let back = g.add(forward, forward); // %1 → %0: cycle!
        let text = to_rec_expr(&g, forward).to_string();
        assert_eq!(
            text,
            "(add (add v0cycle v0cycle) (add v0cycle v0cycle))"
        );
        assert_eq!(back.0, 1);
    }

    /// last_defined picks the highest defined value and skips trailing
    /// params; None for a defs-only-params graph.
    #[test]
    fn last_defined_is_highest_def() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let s = g.add(a, a);
        assert_eq!(last_defined(&g), Some(s));
        let mut empty = Ssa::new();
        empty.new_param();
        empty.new_param();
        assert_eq!(last_defined(&empty), None);
    }
}
