//! SSA Data Flow Graph IR (Module 5).
//!
//! Straight-line single-assignment form that the future tree-sitter
//! frontends (Python via `ast`, C++ via libclang) will LOWER INTO before
//! egg equality saturation (see [`crate::engine`]). Stdlib-only by design:
//! no egg/tokio dependency so the IR can be built and validated anywhere.
//!
//! Dominance note: this is the degenerate straight-line form — a def
//! "dominates" its uses iff the def index precedes the use index and the
//! use actually references it. Real control flow arrives with the CFG
//! extension later; `validate()` still guarantees the soundness core
//! (no use-before-def, no dangling operands).

use std::fmt;

/// An SSA register: dense index into [`Ssa::defs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub u32);

/// Operations producing an SSA value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Const(i64),
    Add(Value, Value),
    Mul(Value, Value),
    Neg(Value),
}

impl Op {
    fn operands(&self) -> Vec<Value> {
        match self {
            Op::Const(_) => vec![],
            Op::Add(a, b) | Op::Mul(a, b) => vec![*a, *b],
            Op::Neg(a) => vec![*a],
        }
    }
}

/// IR validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaError {
    /// An operand references a value defined later (or itself).
    UseBeforeDef { use_idx: u32, def_idx: u32 },
    /// An operand references a value that does not exist.
    UnknownValue(u32),
}

impl fmt::Display for SsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaError::UseBeforeDef { use_idx, def_idx } => {
                write!(f, "use-before-def: value %{use_idx} uses def %{def_idx}")
            }
            SsaError::UnknownValue(v) => write!(f, "unknown value %{v}"),
        }
    }
}

impl std::error::Error for SsaError {}

/// Straight-line SSA data flow graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ssa {
    defs: Vec<Option<Op>>,
    params: Vec<Value>,
}

impl Ssa {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare an input parameter (a def-less value).
    pub fn new_param(&mut self) -> Value {
        let v = Value(self.defs.len() as u32);
        self.defs.push(None);
        self.params.push(v);
        v
    }

    fn push(&mut self, op: Op) -> Value {
        let v = Value(self.defs.len() as u32);
        self.defs.push(Some(op));
        v
    }

    /// `%v = const i`
    pub fn constant(&mut self, v: i64) -> Value {
        self.push(Op::Const(v))
    }

    /// `%v = add a b`
    pub fn add(&mut self, a: Value, b: Value) -> Value {
        self.push(Op::Add(a, b))
    }

    /// `%v = mul a b`
    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        self.push(Op::Mul(a, b))
    }

    /// `%v = neg a`
    pub fn neg(&mut self, a: Value) -> Value {
        self.push(Op::Neg(a))
    }

    /// Total number of values (params + defs) minted so far.
    ///
    /// Stdlib-pure introspection for callers that must distinguish an
    /// in-range param from a dangling id (used by [`crate::ssa_bridge`] to
    /// render out-of-range operands as inert atoms instead of fake params).
    pub fn len(&self) -> usize {
        self.defs.len()
    }

    /// True when no value has been minted yet.
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// The operation defining `v`, if any (None for params).
    pub fn op_of(&self, v: Value) -> Option<&Op> {
        self.defs.get(v.0 as usize).and_then(|d| d.as_ref())
    }

    /// True when `v` is a declared parameter.
    pub fn is_param(&self, v: Value) -> bool {
        (v.0 as usize) < self.defs.len()
            && self.defs[v.0 as usize].is_none()
    }

    /// Values whose operation references `v`.
    pub fn uses(&self, v: Value) -> Vec<Value> {
        self.defs
            .iter()
            .enumerate()
            .filter(|(_, d)| {
                d.as_ref().is_some_and(|op| op.operands().contains(&v))
            })
            .map(|(i, _)| Value(i as u32))
            .collect()
    }

    /// Straight-line dominance: `a` precedes `b` AND `b` uses `a`.
    pub fn dominates(&self, a: Value, b: Value) -> bool {
        a.0 < b.0
            && self
                .op_of(b)
                .is_some_and(|op| op.operands().contains(&a))
    }

    /// Full soundness check: every operand exists and is defined strictly
    /// before its use.
    pub fn validate(&self) -> Result<(), SsaError> {
        for (idx, def) in self.defs.iter().enumerate() {
            let idx = idx as u32;
            if let Some(op) = def {
                for operand in op.operands() {
                    if operand.0 as usize >= self.defs.len() {
                        return Err(SsaError::UnknownValue(operand.0));
                    }
                    if operand.0 >= idx {
                        return Err(SsaError::UseBeforeDef {
                            use_idx: idx,
                            def_idx: operand.0,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Constant-fold evaluation of `v` given parameter values.
    pub fn evaluate(&self, v: Value, params: &[i64]) -> Result<i64, SsaError> {
        match self.op_of(v) {
            None => {
                let ordinal = self
                    .params
                    .iter()
                    .position(|p| *p == v)
                    .ok_or(SsaError::UnknownValue(v.0))?;
                params.get(ordinal).copied().ok_or(SsaError::UnknownValue(v.0))
            }
            Some(op) => match op.clone() {
                Op::Const(c) => Ok(c),
                Op::Add(a, b) => {
                    Ok(self.evaluate(a, params)? + self.evaluate(b, params)?)
                }
                Op::Mul(a, b) => {
                    Ok(self.evaluate(a, params)? * self.evaluate(b, params)?)
                }
                Op::Neg(a) => Ok(-self.evaluate(a, params)?),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// %0 = param, %1 = param, %2 = add %0 %1  →  eval [2, 3] == 5.
    #[test]
    fn add_params_evaluates() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let b = g.new_param();
        let c = g.add(a, b);
        assert!(g.validate().is_ok());
        assert_eq!(g.evaluate(c, &[2, 3]).unwrap(), 5);
    }

    #[test]
    fn nested_ops_evaluate() {
        let mut g = Ssa::new();
        let p = g.new_param();
        let three = g.constant(3);
        let m = g.mul(p, three);
        let n = g.neg(m);
        assert_eq!(g.evaluate(n, &[4]).unwrap(), -12);
    }

    #[test]
    fn uses_reports_referencing_values() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let c = g.constant(1);
        let s = g.add(a, c);
        let mut u = g.uses(a);
        u.sort_by_key(|v| v.0);
        assert_eq!(u, vec![s]);
        assert!(g.uses(c).contains(&s));
    }

    #[test]
    fn validate_ok_on_well_formed() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let b = g.constant(2);
        g.add(a, b);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn use_before_def_is_rejected() {
        let mut g = Ssa::new();
        let a = g.add(Value(2), Value(3)); // forward refs!
        let _x = g.constant(1);
        let _y = g.constant(2);
        let _z = g.constant(3);
        let err = g.validate().unwrap_err();
        assert_eq!(
            err,
            SsaError::UseBeforeDef { use_idx: a.0, def_idx: 2 }
        );
    }

    #[test]
    fn unknown_value_is_rejected() {
        let mut g = Ssa::new();
        let a = g.new_param();
        g.add(a, Value(99));
        assert_eq!(g.validate(), Err(SsaError::UnknownValue(99)));
    }

    #[test]
    fn dominance_is_def_before_use() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let s = g.add(a, a);
        assert!(g.dominates(a, s));
        assert!(!g.dominates(s, a));
        // same index can never dominate (strict)
        assert!(!g.dominates(a, a));
    }

    #[test]
    fn single_assignment_by_construction() {
        let mut g = Ssa::new();
        let a = g.new_param();
        let v1 = g.add(a, a);
        let v2 = g.add(a, a);
        assert_ne!(v1, v2, "every op mints a fresh Value");
        assert_eq!(g.uses(a).len(), 2);
    }
}
