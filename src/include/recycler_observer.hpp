#pragma once

#include "duckdb/optimizer/optimizer_extension.hpp"

#include <mutex>
#include <unordered_map>

namespace duckdb {

//! State shared by every query on this database: how often each subtree
//! fingerprint has been seen so far.
struct RecyclerState : public OptimizerExtensionInfo {
	std::mutex lock;
	std::unordered_map<hash_t, idx_t> times_seen;
	idx_t queries_seen = 0;
};

//! Optimizer extension that changes nothing: it fingerprints every subtree of
//! the optimized plan and, if `SET waddle_log = true`, reports the repeats.
OptimizerExtension CreateRecyclerObserver();

} // namespace duckdb
