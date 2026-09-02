# Ouroboros DFG

![tests](https://github.com/vivek5200/ouroboros-dfg/actions/workflows/tests.yml/badge.svg)

Rust semantic verifier for the Ouroboros v7.1 code refactoring system.

## Module 5: DFG Verification
Extracts SSA-form Data Flow Graphs via mypy/clang frontends and verifies
semantic equivalence using egg equality saturation.

## Constraints
- egg::Runner limits: IterationLimit(5000), TimeLimit(10s), NodeLimit(1_000_000)
- BackoffScheduler { match_limit: 5000, ban_length: 3 }
- Equality saturation runs as async background CPU task, NEVER blocking GPU
- SSA DFG extraction uses mypy (Python) and libclang (C++) frontends

## Setup
```bash
cargo build
cargo test
```
