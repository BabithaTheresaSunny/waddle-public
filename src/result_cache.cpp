#include "result_cache.hpp"

namespace duckdb {

shared_ptr<CachedResult> ResultCache::Lookup(hash_t fingerprint, const string &plan_key) {
	lock_guard<mutex> guard(lock);
	auto entry = entries.find(fingerprint);
	// A matching fingerprint is only a hint; the plans must be identical.
	if (entry == entries.end() || entry->second->plan_key != plan_key) {
		misses++;
		return nullptr;
	}
	hits++;
	entry->second->last_used = ++clock;
	return entry->second;
}

void ResultCache::Insert(hash_t fingerprint, string plan_key, vector<LogicalType> types,
                         shared_ptr<ColumnDataCollection> data) {
	auto bytes = data->AllocationSize();
	lock_guard<mutex> guard(lock);
	if (bytes > budget) {
		return;
	}
	auto existing = entries.find(fingerprint);
	if (existing != entries.end()) {
		if (existing->second->plan_key == plan_key) {
			// We already have this exact result. DuckDB can execute the same
			// operator more than once within one query - a second run may see
			// no rows at all - so the first complete capture is the one we
			// keep, and a later one is never allowed to replace it.
			return;
		}
		// Same fingerprint, different plan: keep the newer result.
		bytes_used -= existing->second->bytes;
		entries.erase(existing);
	}
	EvictUntilFits(bytes);
	auto result = make_shared_ptr<CachedResult>();
	result->fingerprint = fingerprint;
	result->plan_key = std::move(plan_key);
	result->types = std::move(types);
	result->data = std::move(data);
	result->bytes = bytes;
	result->last_used = ++clock;
	entries[fingerprint] = std::move(result);
	bytes_used += bytes;
}

void ResultCache::EvictUntilFits(idx_t incoming_bytes) {
	while (!entries.empty() && bytes_used + incoming_bytes > budget) {
		auto victim = entries.begin();
		for (auto it = entries.begin(); it != entries.end(); it++) {
			if (it->second->last_used < victim->second->last_used) {
				victim = it;
			}
		}
		bytes_used -= victim->second->bytes;
		// The rows themselves live on as long as a running query holds a
		// shared_ptr to them.
		entries.erase(victim);
	}
}

void ResultCache::Clear() {
	lock_guard<mutex> guard(lock);
	entries.clear();
	bytes_used = 0;
}

void ResultCache::SetBudget(idx_t bytes) {
	lock_guard<mutex> guard(lock);
	budget = bytes;
	EvictUntilFits(0);
}

idx_t ResultCache::MaxEntryBytes() {
	lock_guard<mutex> guard(lock);
	return budget / 4;
}

} // namespace duckdb
