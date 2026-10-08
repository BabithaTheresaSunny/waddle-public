#pragma once

#include "result_cache.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

#include <unordered_map>

namespace duckdb {

//! State shared by every query on this database.
struct RecyclerState : public OptimizerExtensionInfo {
	shared_ptr<ResultCache> cache = make_shared_ptr<ResultCache>();
	idx_t queries_seen = 0;

	//! How many earlier queries contained each subtree. Materialising an
	//! intermediate that never comes back is pure overhead, so a subtree is
	//! only captured once it has been seen before. Fingerprints only - a few
	//! bytes per subtree, and nothing that could be mistaken for a result.
	mutex seen_lock;
	std::unordered_map<hash_t, idx_t> subtrees_seen;

	//! Returns how many earlier queries contained this subtree, and counts
	//! this sighting.
	idx_t RecordSighting(hash_t fingerprint) {
		lock_guard<mutex> guard(seen_lock);
		return subtrees_seen[fingerprint]++;
	}
};

//! Optimizer extension that reuses results it has computed before, either of a
//! whole query or of one subtree of a query.
OptimizerExtension CreateRecyclerOptimizer();

} // namespace duckdb
