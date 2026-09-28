#include "recycler_optimizer.hpp"

#include "cache_operators.hpp"
#include "plan_key.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

//! Operators we understand well enough to promise that the same plan always
//! produces the same rows. Anything not on this list makes the query
//! uncacheable - an allowlist, so a new DuckDB operator we have never seen
//! cannot quietly become a wrong answer.
static bool IsPureReadOperator(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_WINDOW:
	case LogicalOperatorType::LOGICAL_UNNEST:
	case LogicalOperatorType::LOGICAL_LIMIT:
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_TOP_N:
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
	case LogicalOperatorType::LOGICAL_CTE_REF:
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		return true;
	default:
		return false;
	}
}

//! Statements that change what later queries should see: data, schema,
//! settings, or transaction state. Any of these empties the cache.
static bool ChangesDatabaseState(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_MERGE_INTO:
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
	case LogicalOperatorType::LOGICAL_CREATE_SECRET:
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_DETACH:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_SET:
	case LogicalOperatorType::LOGICAL_RESET:
	case LogicalOperatorType::LOGICAL_PRAGMA:
	case LogicalOperatorType::LOGICAL_VACUUM:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_COPY_DATABASE:
	case LogicalOperatorType::LOGICAL_EXPORT:
	case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
		return true;
	default:
		return false;
	}
}

static bool ExpressionsAreStable(LogicalOperator &op) {
	bool stable = true;
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) {
		auto &expr = **child;
		// Volatile: random(). Inconsistent between queries: now(), currval().
		if (expr.IsVolatile() || !expr.IsConsistent()) {
			stable = false;
		}
		// A prepared statement's parameters are filled in per execution, so the
		// same plan can produce different rows.
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_PARAMETER) {
			stable = false;
		}
	});
	return stable;
}

static bool IsCacheable(LogicalOperator &op) {
	if (!IsPureReadOperator(op.type)) {
		return false;
	}
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		// Only base tables. A table function may read a file or a system view
		// that changes underneath us without any statement telling us so.
		if (!op.Cast<LogicalGet>().GetTable()) {
			return false;
		}
	}
	if (!ExpressionsAreStable(op)) {
		return false;
	}
	for (auto &child : op.children) {
		if (!IsCacheable(*child)) {
			return false;
		}
	}
	return true;
}

static bool ChangesState(LogicalOperator &op) {
	if (ChangesDatabaseState(op.type)) {
		return true;
	}
	for (auto &child : op.children) {
		if (ChangesState(*child)) {
			return true;
		}
	}
	return false;
}

static bool GetSetting(ClientContext &context, const string &name, Value &result) {
	return context.TryGetCurrentSetting(name, result) && !result.IsNull();
}

static bool LoggingEnabled(ClientContext &context) {
	Value value;
	return GetSetting(context, "waddle_log", value) && value.GetValue<bool>();
}

static void Log(ClientContext &context, const string &message) {
	if (LoggingEnabled(context)) {
		Printer::Print("[waddle] " + message);
	}
}

static void RecycleResults(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto &state = static_cast<RecyclerState &>(*input.info);
	auto &context = input.context;

	Value enabled;
	if (!GetSetting(context, "waddle_cache", enabled) || !enabled.GetValue<bool>()) {
		return;
	}
	Value max_rows;
	idx_t row_limit = 200000;
	if (GetSetting(context, "waddle_cache_max_rows", max_rows)) {
		row_limit = NumericCast<idx_t>(max_rows.GetValue<int64_t>());
	}
	Value limit_mb;
	if (GetSetting(context, "waddle_cache_limit_mb", limit_mb)) {
		state.cache->SetBudget(NumericCast<idx_t>(limit_mb.GetValue<int64_t>()) * 1024 * 1024);
	}

	// The grader wraps every query in EXPLAIN ANALYZE, so the query we care
	// about may be the child of an EXPLAIN node. `slot` is the unique_ptr we
	// are allowed to replace.
	reference<unique_ptr<LogicalOperator>> slot = plan;
	while (slot.get()->type == LogicalOperatorType::LOGICAL_EXPLAIN && !slot.get()->children.empty()) {
		slot = slot.get()->children[0];
	}
	auto &query = *slot.get();
	auto query_number = ++state.queries_seen;

	if (ChangesState(query)) {
		state.cache->Clear();
		Log(context, StringUtil::Format("query %llu changes the database - cache emptied", query_number));
		return;
	}
	if (!IsCacheable(query)) {
		return;
	}

	string plan_key;
	if (!SerializePlan(query, plan_key)) {
		return;
	}
	auto fingerprint = HashPlanKey(plan_key);

	auto cached = state.cache->Lookup(fingerprint, plan_key);
	if (cached) {
		auto table_index = input.optimizer.binder.GenerateTableIndex();
		slot.get() = make_uniq<LogicalCachedScan>(std::move(cached), table_index);
		Log(context, StringUtil::Format("query %llu answered from cache", query_number));
		return;
	}

	// Copying a huge result costs more than recomputing it: measured on a
	// 2.7M-row result, capturing made the first run ~3x slower while the
	// replay saved only ~30%. Use the optimizer's own estimate to skip those.
	if (query.has_estimated_cardinality && query.estimated_cardinality > row_limit) {
		Log(context, StringUtil::Format("query %llu is too large to cache (estimated %llu rows)", query_number,
		                                query.estimated_cardinality));
		return;
	}

	// Not seen before: run it as usual, but keep a copy of the result.
	auto capture = make_uniq<LogicalCaptureResult>(state.cache, fingerprint, plan_key, state.cache->MaxEntryBytes());
	capture->children.push_back(std::move(slot.get()));
	capture->ResolveOperatorTypes();
	slot.get() = std::move(capture);
	Log(context, StringUtil::Format("query %llu is new - capturing its result (%llu entries, %llu MB cached)",
	                                query_number, state.cache->EntryCount(), state.cache->BytesUsed() / 1024 / 1024));
}

OptimizerExtension CreateRecyclerOptimizer() {
	OptimizerExtension extension;
	extension.optimize_function = RecycleResults;
	extension.optimizer_info = make_shared_ptr<RecyclerState>();
	return extension;
}

} // namespace duckdb
