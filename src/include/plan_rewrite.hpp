#pragma once

#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! Points every column reference in `root` that used to come out of a replaced
//! subtree at the cached scan that replaced it.
//!
//! A subtree produces columns under the table indices of the operators inside
//! it. Swapping the subtree for a cached scan retires those indices and
//! introduces one new index, so every expression above the swap would still be
//! asking for columns that no longer exist. `old_bindings` is what the subtree
//! produced, in order, and column i of it becomes column i of the new table.
void RemapColumnBindings(LogicalOperator &root, const vector<ColumnBinding> &old_bindings, idx_t new_table_index);

} // namespace duckdb
