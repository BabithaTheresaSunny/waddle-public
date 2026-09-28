#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <unordered_map>

namespace duckdb {

//! One cached query result: the rows a query produced, plus enough information
//! to be sure a later query really is the same query.
struct CachedResult {
	hash_t fingerprint;
	//! The serialized plan. The fingerprint is only a hash, and two different
	//! plans can hash to the same value, so the bytes themselves decide
	//! whether a later query may reuse this result.
	string plan_key;
	vector<LogicalType> types;
	shared_ptr<ColumnDataCollection> data;
	idx_t bytes;
	//! Logical clock value of the last hit, used to pick an eviction victim.
	idx_t last_used;
};

//! Whole-query result cache, shared by every connection to this database.
class ResultCache {
public:
	//! Returns the cached result for this plan, or nullptr. The caller holds a
	//! shared_ptr, so the rows stay alive even if the entry is evicted while
	//! the query is still running.
	shared_ptr<CachedResult> Lookup(hash_t fingerprint, const string &plan_key);
	//! Stores a result, evicting least-recently-used entries if needed.
	void Insert(hash_t fingerprint, string plan_key, vector<LogicalType> types,
	            shared_ptr<ColumnDataCollection> data);
	//! Drops everything. Called whenever the database changes.
	void Clear();

	void SetBudget(idx_t bytes);
	//! Largest result worth storing: a quarter of the budget, so one big result
	//! can never evict everything else.
	idx_t MaxEntryBytes();

	idx_t Hits() const {
		return hits;
	}
	idx_t Misses() const {
		return misses;
	}
	idx_t BytesUsed() const {
		return bytes_used;
	}
	idx_t EntryCount() const {
		return entries.size();
	}

private:
	//! Caller must hold `lock`.
	void EvictUntilFits(idx_t incoming_bytes);

	mutex lock;
	std::unordered_map<hash_t, shared_ptr<CachedResult>> entries;
	idx_t budget = 512ULL * 1024 * 1024;
	idx_t bytes_used = 0;
	idx_t clock = 0;
	idx_t hits = 0;
	idx_t misses = 0;
};

} // namespace duckdb
