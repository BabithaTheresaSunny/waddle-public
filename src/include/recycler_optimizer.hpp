#pragma once

#include "result_cache.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

namespace duckdb {

//! State shared by every query on this database.
struct RecyclerState : public OptimizerExtensionInfo {
	shared_ptr<ResultCache> cache = make_shared_ptr<ResultCache>();
	idx_t queries_seen = 0;
};

//! Optimizer extension that reuses the result of a query it has run before.
OptimizerExtension CreateRecyclerOptimizer();

} // namespace duckdb
