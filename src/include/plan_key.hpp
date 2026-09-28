#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! Serializes a plan into bytes that describe it completely, and returns false
//! if DuckDB cannot serialize some part of it.
//!
//! We deliberately do NOT key the cache on the plan's printed form. Several
//! operators leave fields out of EXPLAIN: LogicalLimit and LogicalTopN do not
//! print their limit or offset at all, so "LIMIT 10000 OFFSET 0" and
//! "LIMIT 10000 OFFSET 10000" print identically and would share a cache entry.
//! Serialization is the format DuckDB itself relies on to reconstruct a plan,
//! so if two plans serialize to the same bytes they really are the same plan.
bool SerializePlan(const LogicalOperator &op, string &result);

//! Hash of a serialized plan, used to find candidates. The bytes themselves
//! still decide whether a candidate is a match.
hash_t HashPlanKey(const string &key);

//! One subtree of a plan, identified by its serialized form. Used to analyse
//! how much work repeats across a workload.
struct SubtreeInfo {
	hash_t fingerprint;
	const LogicalOperator *op;
	idx_t num_operators;
};

//! Fingerprints every subtree of `root`, children before parents. Subtrees that
//! cannot be serialized are left out.
vector<SubtreeInfo> FingerprintSubtrees(const LogicalOperator &root);

} // namespace duckdb
