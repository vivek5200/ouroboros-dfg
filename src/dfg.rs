//! Module 5: Data Flow Graph extraction.
//!
//! Extracts SSA-form Data Flow Graphs from source code using:
//! - mypy frontend (for Python code)
//! - libclang frontend (for C++ code)

use serde::{Deserialize, Serialize};

/// A node in the SSA-form Data Flow Graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DfgNode {
    pub id: usize,
    pub op: String,
    pub operands: Vec<usize>,
}

/// A complete Data Flow Graph in SSA form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowGraph {
    pub nodes: Vec<DfgNode>,
    pub source_file: String,
}

impl DataFlowGraph {
    /// Create a new empty DFG.
    pub fn new(source_file: &str) -> Self {
        Self {
            nodes: Vec::new(),
            source_file: source_file.to_string(),
        }
    }

    /// Add a node to the DFG.
    pub fn add_node(&mut self, op: &str, operands: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(DfgNode {
            id,
            op: op.to_string(),
            operands,
        });
        id
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dfg_is_empty() {
        let dfg = DataFlowGraph::new("test.py");
        assert_eq!(dfg.node_count(), 0);
        assert_eq!(dfg.source_file, "test.py");
    }

    #[test]
    fn test_add_node() {
        let mut dfg = DataFlowGraph::new("test.py");
        let id = dfg.add_node("add", vec![0, 1]);
        assert_eq!(id, 0);
        assert_eq!(dfg.node_count(), 1);
    }

    #[test]
    fn test_node_operands() {
        let mut dfg = DataFlowGraph::new("test.py");
        dfg.add_node("const", vec![]);
        dfg.add_node("const", vec![]);
        let add_id = dfg.add_node("add", vec![0, 1]);
        assert_eq!(dfg.nodes[add_id].operands, vec![0, 1]);
    }
}
