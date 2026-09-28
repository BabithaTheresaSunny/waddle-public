#pragma once

#include "result_cache.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

//! Sits at the top of a plan we want to remember. It passes every row through
//! unchanged and keeps a copy; when the query finishes, the copy goes into the
//! cache. If the result grows past `max_bytes` it stops copying and the query
//! simply is not cached.
class LogicalCaptureResult : public LogicalExtensionOperator {
public:
	LogicalCaptureResult(shared_ptr<ResultCache> cache, hash_t fingerprint, string plan_key, idx_t max_bytes);

	shared_ptr<ResultCache> cache;
	hash_t fingerprint;
	string plan_key;
	idx_t max_bytes;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	string GetName() const override;
	vector<ColumnBinding> GetColumnBindings() override;
	string GetExtensionName() const override;

protected:
	void ResolveTypes() override;
};

//! Replaces a whole plan whose result we already have. It produces the cached
//! rows and reads nothing from storage.
class LogicalCachedScan : public LogicalExtensionOperator {
public:
	LogicalCachedScan(shared_ptr<CachedResult> result, idx_t table_index);

	shared_ptr<CachedResult> result;
	idx_t table_index;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	string GetName() const override;
	vector<ColumnBinding> GetColumnBindings() override;
	vector<idx_t> GetTableIndex() const override;
	string GetExtensionName() const override;

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
