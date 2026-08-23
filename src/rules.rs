//! Module 5 rewrite systems (paper §7.2): math + boolean rule sets.
//!
//! Bidirectional rewrite rules — De Morgan's Laws, algebraic identities,
//! double-negation. Equivalence is proven iff both roots collapse into one
//! e-class (checked via [`egg::EGraph::equivs`], engine.rs); the SRE
//! telemetry layer tracks the resulting green/yellow ratio (§7.4).
//!
//! LANGUAGE DESIGN: both families share [`SymbolLang`] rather than defining
//! a second `define_language!` enum. A `Language` type parameter admits one
//! op enum per e-graph, so a separate Boolean enum would make mixed
//! expressions unrepresentable and force a second Runner plumbing path.
//! `SymbolLang` treats every operator as an opaque symbol, so `not`/`and`/`or`
//! coexist with `+`/`*` in one e-graph and cross-family pairs simply stay in
//! distinct e-classes (→ Unproven) instead of failing to parse.
//!
//! EGG API SOURCES (verified against local crate source, not guessed):
//! - `~/.cargo/registry/src/*/egg-0.9.5/src/macros.rs:282`
//!     - `macro_rules! rewrite`. The `<=>` arm (pattern at line 295) does
//!       NOT produce one rewrite: it evaluates to
//!       `vec![rewrite!(name; lhs => rhs), rewrite!(name2; rhs => lhs)]`
//!       where `name2 = name + "-rev"` (lines 299–303). Hence every
//!       bidirectional rule below goes through `Vec::extend`, never `push`.
//!     - Variables use the `?x` s-expression syntax shown in the macro's own
//!       doc examples (`"(+ ?a 0)" => "?a"`, macros.rs:265–269); the applier
//!       rejects unbound vars (`Rewrite::new` → "refers to unbound var",
//!       src/rewrite.rs:52–55).
//!     - egg's own doc examples state the algebraic identities as `<=>`
//!       (macros.rs:259–260). We deliberately keep them ONE-directional
//!       (`x+0 => x`, `x*1 => x`) per the Module 5 task spec: saturation
//!       still proves `(f x 0)` ≡ `x` because the e-graph retains both the
//!       original node and its rewrite, while the directed form documents
//!       these as simplifications.

use egg::{rewrite as rw, Rewrite, SymbolLang};

/// Math rule system: AC-style addition plus the algebraic identities
/// `x+0 => x` and `x*1 => x`.
///
/// Commutativity needs only ONE direction: child-swap is involutive, so
/// `"(+ ?a ?b)" => "(+ ?b ?a)"` already realizes the full `<->` pair.
/// Associativity genuinely requires both directions.
pub fn math_rules() -> Vec<Rewrite<SymbolLang, ()>> {
    vec![
        rw!("commute-add"; "(+ ?a ?b)"          => "(+ ?b ?a)"),
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))"     => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-add-flip"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        // Algebraic simplifications (one-directional by design; see module docs).
        rw!("add-zero"; "(+ ?a 0)" => "?a"),
        rw!("mul-one"; "(* ?a 1)" => "?a"),
    ]
}

/// Boolean rule system over ops `not` (1 arg), `and`/`or` (2 args).
///
/// Both De Morgan laws are TRUE bidirectional pairs: each direction is a
/// sound equivalence on classical logic and neither subsumes the other, so
/// they expand to two rewrites each via `<=>` (macros.rs:295–303).
/// Double-negation is kept ONE-directional (`not(not(a)) => a`): it is a
/// simplification, and the reverse would only add noise nodes during
/// saturation without enabling any proof the forward set cannot reach.
pub fn boolean_rules() -> Vec<Rewrite<SymbolLang, ()>> {
    let mut rules: Vec<Rewrite<SymbolLang, ()>> = Vec::new();
    // De Morgan #1: not(and(a,b)) <=> or(not(a), not(b))
    rules.extend(rw!("de-morgan-not-and";
        "(not (and ?a ?b))" <=> "(or (not ?a) (not ?b))"));
    // De Morgan #2: not(or(a,b)) <=> and(not(a), not(b))
    rules.extend(rw!("de-morgan-not-or";
        "(not (or ?a ?b))" <=> "(and (not ?a) (not ?b))"));
    // Double negation elimination.
    rules.push(rw!("double-negation"; "(not (not ?a))" => "?a"));
    rules
}

/// The union of [`math_rules`] and [`boolean_rules`] — the rule set behind
/// [`crate::engine::EggEngine::verify`].
///
/// Merging is safe because the families are disjoint in their root symbols:
/// no rule can map a `+`/`*`-rooted term to an `and`/`or`-rooted term, so a
/// math pair can only merge through math rules (boolean likewise), while
/// sharing ONE e-graph lets callers submit mixed jobs under a single LAW-
/// bounded runner instead of maintaining two engines.
pub fn all_rules() -> Vec<Rewrite<SymbolLang, ()>> {
    let mut rules = math_rules();
    rules.extend(boolean_rules());
    rules
}

/// SSA-dialect mirror of [`math_rules`], for graphs lowered by
/// [`crate::ssa_bridge::to_rec_expr`].
///
/// WHY A SECOND SPELLING: the bridge is pinned to IR op names `add`/`mul`/
/// `neg` (task contract; params are `p<N>`, constants are bare literals).
/// `SymbolLang` treats operators as opaque symbols (rules.rs module docs),
/// so `(add p0 p1)` matches NOTHING in [`math_rules`] — its patterns spell
/// the operator `+`. These twins carry the identical §7.2 identities
/// (commutativity, both associativity directions, `x+0 => x`, `x*1 => x`)
/// under the IR spelling. Same one-directionality choices as
/// [`math_rules`]: child-swap realizes full commutativity; identities stay
/// directed simplifications.
pub fn ssa_math_rules() -> Vec<Rewrite<SymbolLang, ()>> {
    vec![
        rw!("ssa-commute-add"; "(add ?a ?b)"            => "(add ?b ?a)"),
        rw!("ssa-assoc-add"; "(add ?a (add ?b ?c))"     => "(add (add ?a ?b) ?c)"),
        rw!("ssa-assoc-add-flip"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        // Algebraic simplifications over emitted integer literals.
        rw!("ssa-add-zero"; "(add ?a 0)" => "?a"),
        rw!("ssa-mul-one"; "(mul ?a 1)" => "?a"),
    ]
}

/// The union behind SSA verification
/// ([`crate::engine::EggEngine::verify_ssa`]): [`ssa_math_rules`] plus
/// [`boolean_rules`]. Disjointness argument identical to [`all_rules`]:
/// `add`/`mul` roots cannot cross into `and`/`or` roots, so the shared
/// e-graph stays safe and mixed jobs run under ONE LAW-bounded runner.
pub fn ssa_all_rules() -> Vec<Rewrite<SymbolLang, ()>> {
    let mut rules = ssa_math_rules();
    rules.extend(boolean_rules());
    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LAW of the `<=>` expansion: each bidirectional pair yields exactly
    /// two rewrites (macros.rs:295–303), so 2 pairs + 1 directed rule = 5.
    #[test]
    fn boolean_rules_expand_de_morgan_pairs_both_ways() {
        let rules = boolean_rules();
        assert_eq!(rules.len(), 5);
        let names: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
        for expected in [
            "de-morgan-not-and",
            "de-morgan-not-and-rev",
            "de-morgan-not-or",
            "de-morgan-not-or-rev",
            "double-negation",
        ] {
            assert!(names.contains(&expected), "missing {expected}: {names:?}");
        }
    }

    #[test]
    fn math_rules_carry_algebraic_identities() {
        let rules = math_rules();
        assert_eq!(rules.len(), 5);
        let names: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"add-zero"), "{names:?}");
        assert!(names.contains(&"mul-one"), "{names:?}");
    }

    #[test]
    fn all_rules_is_exact_disjoint_union() {
        assert_eq!(all_rules().len(), math_rules().len() + boolean_rules().len());
    }

    /// The SSA dialect mirrors the math system rule-for-rule (5 rewrites:
    /// commute + both assoc directions + two directed identities).
    #[test]
    fn ssa_math_rules_mirror_math_rules() {
        let rules = ssa_math_rules();
        assert_eq!(rules.len(), 5);
        let names: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
        for expected in [
            "ssa-commute-add",
            "ssa-assoc-add",
            "ssa-assoc-add-flip",
            "ssa-add-zero",
            "ssa-mul-one",
        ] {
            assert!(names.contains(&expected), "missing {expected}: {names:?}");
        }
    }

    #[test]
    fn ssa_all_rules_is_exact_disjoint_union() {
        assert_eq!(
            ssa_all_rules().len(),
            ssa_math_rules().len() + boolean_rules().len()
        );
        // Dialects stay disjoint: no SSA-spelled rule shares a name with the
        // `+`-spelled system, so the two unions cannot cross-fire.
        let math: Vec<&str> = math_rules().iter().map(|r| r.name.as_str()).collect();
        let ssa: Vec<&str> = ssa_math_rules().iter().map(|r| r.name.as_str()).collect();
        assert!(math.iter().all(|n| !ssa.contains(n)));
    }
}
