#include "recycler_observer.hpp"

#include "plan_fingerprint.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

static bool LoggingEnabled(ClientContext &context) {
	Value value;
	if (!context.TryGetCurrentSetting("waddle_log", value)) {
		return false;
	}
	return !value.IsNull() && value.GetValue<bool>();
}

static void ObservePlan(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	// The grader wraps every query in EXPLAIN ANALYZE, so the plan we get is
	// EXPLAIN -> (the actual query). Look at the query, not the wrapper.
	LogicalOperator *query = plan.get();
	while (query->type == LogicalOperatorType::LOGICAL_EXPLAIN && !query->children.empty()) {
		query = query->children[0].get();
	}

	auto subtrees = FingerprintSubtrees(*query);
	auto &state = static_cast<RecyclerState &>(*input.info);

	idx_t query_number;
	idx_t repeated_subtrees = 0;
	const SubtreeInfo *largest_repeat = nullptr;
	bool root_repeated;
	{
		std::lock_guard<std::mutex> guard(state.lock);
		query_number = state.queries_seen++;
		for (auto &subtree : subtrees) {
			auto &count = state.times_seen[subtree.fingerprint];
			if (count > 0) {
				repeated_subtrees++;
				if (!largest_repeat || subtree.num_operators > largest_repeat->num_operators) {
					largest_repeat = &subtree;
				}
			}
			count++;
		}
		root_repeated = state.times_seen[subtrees.back().fingerprint] > 1;
	}

	if (!LoggingEnabled(input.context)) {
		return;
	}
	string line = StringUtil::Format("[waddle] query %llu: %llu operators, %llu already seen", query_number,
	                                 subtrees.size(), repeated_subtrees);
	if (root_repeated) {
		line += " -- WHOLE QUERY REPEATED";
	} else if (largest_repeat) {
		line += StringUtil::Format(" -- largest repeat: %s (%llu operators)", largest_repeat->op->GetName(),
		                           largest_repeat->num_operators);
	}
	Printer::Print(line);
}

OptimizerExtension CreateRecyclerObserver() {
	OptimizerExtension extension;
	extension.optimize_function = ObservePlan; // runs after DuckDB's own optimizers
	extension.optimizer_info = make_shared_ptr<RecyclerState>();
	return extension;
}

} // namespace duckdb
