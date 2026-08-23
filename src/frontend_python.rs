//! Module 5 Python frontend: lowers a constrained but REAL subset of Python
//! source into the SSA IR ([`crate::ssa`]) using tree-sitter.
//!
//! SUPPORTED SUBSET (everything else is a [`LoweringError`] saying
//! "unsupported"):
//! - integer literals (including `1_000` digit separators) → [`Op::Const`]
//! - `name = expr` assignment; REBINDING MINTS A FRESH VALUE (SSA): the old
//!   binding stays live in the graph while the env moves to the new value
//! - binary `+` / `*` → [`Op::Add`] / [`Op::Mul`]; binary `-` desugars
//!   exactly to `add(lhs, neg(rhs))` because the IR has no Sub opcode
//! - unary `-` → [`Op::Neg`] (unary `+` is the identity)
//! - parenthesized sub-expressions
//! - multiple statements extend ONE graph in source order; bare
//!   expression statements record their value as [`Lowered::last`]
//!
//! NAME POLICY (documented per task): an identifier never seen before on a
//! right-hand side is treated as an INPUT PARAMETER via
//! [`Ssa::new_param`] — the first occurrence mints the parameter, later
//! occurrences reuse that same Value. A name already bound by an earlier
//! assignment is NOT a parameter (its current binding is used instead).
//!
//! STRICTNESS NOTES: comments, strings, floats, augmented/chained
//! assignment, defs/classes/if/while/etc. are all rejected with an
//! "unsupported construct `<kind>`" message rather than silently dropped;
//! any tree-sitter parse error (ERROR/MISSING node) is rejected upfront as
//! a syntax error.
//!
//! TREE-SITTER API SOURCES (verified against downloaded crate sources,
//! versions resolved by cargo — these APIs changed across 0.20→0.25):
//! - `~/.cargo/registry/src/*/tree-sitter-0.25.10/binding_rust/lib.rs:650`
//!     - `Parser::new()` (ts_parser_new).
//! - `…/tree-sitter-0.25.10/binding_rust/lib.rs:666`
//!     - `Parser::set_language(&mut self, language: &Language) ->
//!       Result<(), LanguageError>` — takes a REF in 0.25 (older releases
//!       took the language by value / returned bool).
//! - `…/tree-sitter-0.25.10/binding_rust/lib.rs:789`
//!     - `Parser::parse(&mut self, text: impl AsRef<[u8]>,
//!       old_tree: Option<&Tree>) -> Option<Tree>`.
//! - `…/tree-sitter-0.25.10/binding_rust/lib.rs:418–421`
//!     - `Language::new(builder: LanguageFn) -> Self`; `:614`
//!       `impl From<LanguageFn> for Language` (the `.into()` used below).
//! - `…/tree-sitter-python-0.23.6/bindings/rust/lib.rs:27`
//!     - `pub const LANGUAGE: LanguageFn` — modern grammar crates expose a
//!       `LANGUAGE` const (NOT a `language() -> Language` function like
//!       pre-0.23 crates); its doctest (`lib.rs:14–21`) is the exact setup
//!       pattern used here: `parser.set_language(&LANGUAGE.into())`.
//! - Grammar shapes checked against
//!   `…/tree-sitter-python-0.23.6/src/node-types.json`: `assignment` has
//!   fields `left`/`right`; `binary_operator` has `left`/`operator`/`right`;
//!   `unary_operator` has `argument`; parenthesized expressions surface as
//!   `parenthesized_expression` wrapper nodes.

use std::collections::HashMap;
use std::fmt;

use tree_sitter::{Node, Parser};

use crate::ssa::{Ssa, Value};

/// Frontend failure: an unsupported construct, a syntax error, or an
/// out-of-range integer literal. `message` is human-readable and always
/// names the offending construct kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweringError {
    pub message: String,
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "python frontend: {}", self.message)
    }
}

impl std::error::Error for LoweringError {}

fn err(message: impl Into<String>) -> LoweringError {
    LoweringError { message: message.into() }
}

/// Result of lowering a whole module: the accumulated SSA graph plus the
/// value of the LAST bare expression statement, if there was one
/// (assignments do not disturb `last`).
#[derive(Debug, Clone)]
pub struct Lowered {
    pub ssa: Ssa,
    pub last: Option<Value>,
    /// Final variable → Value environment (private; read via
    /// [`Lowered::value_of`]). Rebinds leave only the freshest Value here.
    env: HashMap<String, Value>,
}

impl Lowered {
    /// The Value currently bound to `name`, if any.
    pub fn value_of(&self, name: &str) -> Option<Value> {
        self.env.get(name).copied()
    }
}

/// Parse and lower `source` (Python) into the SSA IR, returning the graph.
///
/// Convenience wrapper over [`lower_module`] for callers that do not care
/// about the last expression-statement value.
pub fn lower(source: &str) -> Result<Ssa, LoweringError> {
    lower_module(source).map(|l| l.ssa)
}

/// Full-form entry point: lower `source` into [`Lowered`] — the SSA graph
/// extended across ALL top-level statements, plus the value of the last
/// bare expression statement (see module docs for the supported subset).
pub fn lower_module(source: &str) -> Result<Lowered, LoweringError> {
    // Parser/Language setup verified per module docs:
    // Parser::new (binding_rust/lib.rs:650), Language::new(LanguageFn)
    // (binding_rust/lib.rs:418–421), grammar-side `LANGUAGE` const
    // (tree-sitter-python bindings/rust/lib.rs:27).
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter::Language::from(
            tree_sitter_python::LANGUAGE,
        ))
        .map_err(|e| err(format!("failed to load python grammar: {e}")))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| err("internal: python parser returned no tree"))?;

    let root = tree.root_node();
    if root.has_error() {
        // binding_rust/lib.rs:1674 — true iff any ERROR/MISSING node exists.
        return Err(err(format!(
            "syntax error at byte {} (input is outside the supported subset)",
            root.start_byte()
        )));
    }
    if root.kind() != "module" {
        return Err(err(format!(
            "unsupported construct: expected a module, found `{}`",
            root.kind()
        )));
    }

    let bytes = source.as_bytes();
    let mut ssa = Ssa::new();
    let mut env: HashMap<String, Value> = HashMap::new();
    let mut last: Option<Value> = None;

    for i in 0..root.named_child_count() {
        let stmt = root.named_child(i).expect("i < named_child_count");
        match stmt.kind() {
            // NOTE (verified via root_node().to_sexp() on 0.23.6): simple
            // statements — including assignments — are WRAPPED in
            // expression_statement nodes (`(expression_statement
            // (assignment ...))`), so dispatch looks through the wrapper.
            "expression_statement" => {
                let inner = sole_named_child(stmt, bytes)?;
                match inner.kind() {
                    "assignment" => {
                        handle_assignment(&mut ssa, inner, &mut env, bytes)?;
                    }
                    "augmented_assignment" => {
                        return Err(err(
                            "unsupported construct: augmented assignment \
                             (`+=` etc.; supported: plain `=` only)",
                        ));
                    }
                    _ => {
                        let v = lower_expr(&mut ssa, inner, &mut env, bytes)?;
                        last = Some(v);
                    }
                }
            }
            // Some grammar versions surface bare assignment statements at
            // module level; accept both shapes for forward compatibility.
            "assignment" => {
                handle_assignment(&mut ssa, stmt, &mut env, bytes)?;
            }
            other => {
                return Err(err(format!(
                    "unsupported construct `{other}` (supported: assignments, \
                     integer arithmetic with + - *, unary minus, parentheses)"
                )))
            }
        }
    }

    Ok(Lowered { ssa, last, env })
}

/// Lower one `name = expr` assignment node (SSA rebinding semantics).
fn handle_assignment(
    ssa: &mut Ssa,
    node: Node<'_>,
    env: &mut HashMap<String, Value>,
    bytes: &[u8],
) -> Result<(), LoweringError> {
    let left = node
        .child_by_field_name("left")
        .ok_or_else(|| err("malformed assignment: missing `left`"))?;
    let right = node
        .child_by_field_name("right")
        .ok_or_else(|| err("malformed assignment: missing `right`"))?;
    if left.kind() != "identifier" {
        return Err(err(format!(
            "unsupported assignment target `{}` (only plain names)",
            left.kind()
        )));
    }
    // Chained `a = b = …` nests another assignment on the right.
    if matches!(right.kind(), "assignment" | "augmented_assignment") {
        return Err(err(format!(
            "unsupported construct: chained/augmented assignment (`{}`)",
            right.kind()
        )));
    }
    let name =
        left.utf8_text(bytes).map_err(|e| err(format!("invalid identifier encoding: {e}")))?;
    // SSA discipline: evaluate the RHS against the OLD env FIRST, then move
    // the binding to the freshly minted Value.
    let v = lower_expr(ssa, right, env, bytes)?;
    env.insert(name.to_string(), v);
    Ok(())
}

/// Lower one expression subtree to a Value.
fn lower_expr<'t>(
    ssa: &mut Ssa,
    node: Node<'t>,
    env: &mut HashMap<String, Value>,
    bytes: &'t [u8],
) -> Result<Value, LoweringError> {
    match node.kind() {
        // Integer literals (digit separators allowed by the grammar).
        "integer" => {
            let text = node.utf8_text(bytes).map_err(|e| {
                err(format!("invalid integer literal encoding: {e}"))
            })?;
            let cleaned = text.replace('_', "");
            cleaned.parse::<i64>().map(|c| ssa.constant(c)).map_err(|_| {
                err(format!(
                    "integer literal `{text}` out of i64 range"
                ))
            })
        }
        // Names never seen before become parameters (module-docs policy);
        // known names reuse their current binding.
        "identifier" => {
            let name = node.utf8_text(bytes).map_err(|e| {
                err(format!("invalid identifier encoding: {e}"))
            })?;
            if let Some(v) = env.get(name) {
                Ok(*v)
            } else {
                let p = ssa.new_param();
                env.insert(name.to_string(), p);
                Ok(p)
            }
        }
        // `(expr)` — recurse through the wrapper to the single inner expr.
        "parenthesized_expression" => {
            let inner = sole_named_child(node, bytes)?;
            lower_expr(ssa, inner, env, bytes)
        }
        "binary_operator" => {
            let left = node.child_by_field_name("left").ok_or_else(|| {
                err("malformed binary_operator: missing `left`")
            })?;
            let op = node.child_by_field_name("operator").ok_or_else(|| {
                err("malformed binary_operator: missing `operator`")
            })?;
            let right = node.child_by_field_name("right").ok_or_else(|| {
                err("malformed binary_operator: missing `right`")
            })?;
            let l = lower_expr(ssa, left, env, bytes)?;
            let r = lower_expr(ssa, right, env, bytes)?;
            match op.kind() {
                "+" => Ok(ssa.add(l, r)),
                // No Sub opcode in the IR: `a - b` ≡ add(a, neg(b)).
                "-" => {
                    let nr = ssa.neg(r);
                    Ok(ssa.add(l, nr))
                }
                "*" => Ok(ssa.mul(l, r)),
                other => Err(err(format!(
                    "unsupported binary operator `{other}` (supported: + - *)"
                ))),
            }
        }
        "unary_operator" => {
            let argument = node.child_by_field_name("argument").ok_or_else(|| {
                err("malformed unary_operator: missing `argument`")
            })?;
            let op = node.child_by_field_name("operator").ok_or_else(|| {
                err("malformed unary_operator: missing `operator`")
            })?;
            let a = lower_expr(ssa, argument, env, bytes)?;
            match op.kind() {
                "-" => Ok(ssa.neg(a)),
                "+" => Ok(a),
                other => Err(err(format!(
                    "unsupported unary operator `{other}` (supported: -)"
                ))),
            }
        }
        other => Err(err(format!(
            "unsupported construct `{other}` in expression position \
             (supported: integers, names, parentheses, + - *, unary -)"
        ))),
    }
}

/// Exactly-one-named-child helper (expression statements and parentheses).
fn sole_named_child<'t>(
    node: Node<'t>,
    bytes: &'t [u8],
) -> Result<Node<'t>, LoweringError> {
    match node.named_child_count() {
        1 => Ok(node.named_child(0).expect("count == 1")),
        n => Err(err(format!(
            "unsupported construct `{}` with {n} sub-expressions near byte {} \
             (expected exactly one)",
            node.kind(),
            first_byte(node, bytes)
        ))),
    }
}

fn first_byte(node: Node<'_>, _bytes: &[u8]) -> usize {
    node.start_byte()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::Op;

    /// Required test 1: `x = 1 + 2` lowers; evaluate(x) == 3 (no params).
    #[test]
    fn assign_add_literal_lowers_and_evaluates() {
        let l = lower_module("x = 1 + 2").unwrap();
        let x = l.value_of("x").expect("x bound");
        l.ssa.validate().unwrap();
        assert_eq!(l.ssa.evaluate(x, &[]).unwrap(), 3);
        // Plain assignment is not an expression statement: no `last`.
        assert_eq!(l.last, None);
    }

    /// Required test 2: `a` is ASSIGNED constant 2, not a param, so `b`
    /// evaluates with zero parameters → 6.
    #[test]
    fn assigned_names_are_not_params() {
        let l = lower_module("a = 2\nb = a * 3").unwrap();
        let b = l.value_of("b").unwrap();
        assert_eq!(l.ssa.evaluate(b, &[]).unwrap(), 6);
        // b's operand chain contains no params at all.
        assert!(!l.ssa.is_param(b));
        if let Some(Op::Mul(a, c)) = l.ssa.op_of(b) {
            assert_eq!(l.ssa.evaluate(*a, &[]).unwrap(), 2);
            assert_eq!(l.ssa.op_of(*c), Some(&Op::Const(3)));
        } else {
            panic!("b should be a mul");
        }
    }

    /// Required test 3: `x = 1; x = x + 1` mints TWO distinct Values for x;
    /// the second is add(first, const 1); validate() ok.
    #[test]
    fn rebind_mints_fresh_ssa_value() {
        let l = lower_module("x = 1\nx = x + 1").unwrap();
        l.ssa.validate().unwrap();
        let x2 = l.value_of("x").unwrap();
        // Deterministic build order: %0 = const 1 (first x),
        // %1 = const 1 (rhs literal), %2 = add %0 %1 (second x).
        assert_ne!(x2, Value(0));
        assert_eq!(l.ssa.op_of(Value(0)), Some(&Op::Const(1)));
        assert_eq!(l.ssa.op_of(x2), Some(&Op::Add(Value(0), Value(1))));
        assert_eq!(l.ssa.evaluate(x2, &[]).unwrap(), 2);
    }

    /// Required test 4: function definitions are rejected as unsupported.
    #[test]
    fn function_def_is_unsupported() {
        let e = lower("def f():\n  pass").unwrap_err();
        assert!(e.message.contains("unsupported"), "got: {e}");
        assert!(e.message.contains("function_definition"));
    }

    /// Required test 5: `y = -(x + 2)` with unseen `x` as param →
    /// neg(add(param, const 2)); evaluate([5]) == -7.
    #[test]
    fn unary_minus_over_param_addition() {
        let l = lower_module("y = -(x + 2)").unwrap();
        l.ssa.validate().unwrap();
        let y = l.value_of("y").unwrap();
        let (px, c2) = match l.ssa.op_of(y) {
            Some(Op::Neg(inner)) => match l.ssa.op_of(*inner) {
                Some(Op::Add(a, b)) => (*a, *b),
                other => panic!("inner should be add, got {other:?}"),
            },
            other => panic!("y should be neg, got {other:?}"),
        };
        assert!(l.ssa.is_param(px));
        assert_eq!(l.ssa.op_of(c2), Some(&Op::Const(2)));
        assert_eq!(l.ssa.evaluate(y, &[5]).unwrap(), -7);
    }

    /// Binary minus desugars exactly to add(lhs, neg(rhs)).
    #[test]
    fn binary_minus_desugars_to_add_neg() {
        let l = lower_module("z = a - b").unwrap();
        let z = l.value_of("z").unwrap();
        l.ssa.validate().unwrap();
        assert_eq!(l.ssa.evaluate(z, &[10, 3]).unwrap(), 7);
    }

    /// Parentheses group: `(1 + 2) * 3` == 9.
    #[test]
    fn parenthesized_grouping_respected() {
        let l = lower_module("r = (1 + 2) * 3").unwrap();
        let r = l.value_of("r").unwrap();
        assert_eq!(l.ssa.evaluate(r, &[]).unwrap(), 9);
    }

    /// Bare expression statements set `last`; later assignments don't.
    #[test]
    fn last_tracks_final_expression_statement() {
        let l = lower_module("x = 5\nx + 1").unwrap();
        let last = l.last.expect("expression statement present");
        assert_eq!(l.ssa.evaluate(last, &[]).unwrap(), 6);
        assert_eq!(l.value_of("x").is_some(), true);

        let none = lower_module("x = 5").unwrap();
        assert_eq!(none.last, None);
    }

    /// One unseen name used twice shares a SINGLE parameter Value.
    #[test]
    fn repeated_unknown_name_is_one_param() {
        let l = lower_module("y = q + q").unwrap();
        let y = l.value_of("y").unwrap();
        match l.ssa.op_of(y) {
            Some(Op::Add(a, b)) => {
                assert_eq!(a, b, "both occurrences reuse the same param");
                assert!(l.ssa.is_param(*a));
            }
            other => panic!("expected add, got {other:?}"),
        }
        assert_eq!(l.ssa.evaluate(y, &[4]).unwrap(), 8);
    }

    /// Digit separators in literals (`1_000`) parse to 1000.
    #[test]
    fn underscored_integer_literal() {
        let l = lower_module("n = 1_000").unwrap();
        let n = l.value_of("n").unwrap();
        assert_eq!(l.ssa.op_of(n), Some(&Op::Const(1000)));
    }

    /// Strings, floats, augmented and chained assignment all fail loudly.
    #[test]
    fn unsupported_expressions_fail_loudly() {
        for src in [
            "s = \"hello\"",
            "f = 1.5",
            "x = 1\nx += 2",
            "a = b = 1",
            "if x:\n  pass",
            "import os",
            "x = 1 // 2",
            "x = ~y",
        ] {
            let e = lower(src).unwrap_err();
            assert!(
                e.message.contains("unsupported")
                    || e.message.contains("syntax error"),
                "{src:?} → unexpected error: {e}"
            );
        }
    }

    /// Truncated/garbage input hits the parse-error guard, not a panic.
    #[test]
    fn syntax_error_is_reported() {
        let e = lower("x = ").unwrap_err();
        assert!(e.message.contains("syntax error"), "got: {e}");
        // Display includes the frontend prefix.
        assert!(e.to_string().starts_with("python frontend:"));
    }

    /// `lower()` (graph-only convenience) agrees with `lower_module()`.
    #[test]
    fn lower_convenience_matches_lower_module_graph() {
        let g = lower("w = 2 * (3 + 4)").unwrap();
        let l = lower_module("w = 2 * (3 + 4)").unwrap();
        assert_eq!(g, l.ssa);
        assert_eq!(g.validate(), Ok(()));
        let w = l.value_of("w").unwrap();
        assert_eq!(g.evaluate(w, &[]).unwrap(), 14);
    }
}
