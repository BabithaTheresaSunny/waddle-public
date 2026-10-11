#include "recycler_optimizer.hpp"

#include "cache_operators.hpp"
#include "plan_key.hpp"
#include "plan_rewrite.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// What may be cached
//===--------------------------------------------------------------------===//

//! Operators we understand well enough to promise that the same plan always
//! produces the same rows. Anything not on this list makes the plan
//! uncacheable - an allowlist, so a DuckDB operator we have never seen cannot
//! quietly become a wrong answer.
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

//! Operators that read something produced elsewhere in the same plan. A
//! subtree containing one of these is not self-contained: on its own it would
//! have nothing to read from, so it can never be cached or replaced by itself.
static bool DependsOnSurroundingPlan(LogicalOperatorType type) {
	return type == LogicalOperatorType::LOGICAL_DELIM_GET || type == LogicalOperatorType::LOGICAL_CTE_REF;
}

//! Regions of a plan we leave alone when working below the root.
//!
//! LIMIT without an ordering picks an arbitrary set of rows, so the answer
//! depends on the order its input arrives in. Replacing part of that input with
//! cached rows would fix an order that a parallel run never guarantees, and the
//! rows chosen could then differ from the ones vanilla DuckDB returned.
//! Delimited joins and CTE bodies are tied by table index to operators outside
//! the subtree, so swapping a piece of them is not a local change.
static bool BlocksDescent(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_LIMIT:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
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

static bool IsCacheable(LogicalOperator &op, bool allow_dependent) {
	if (!IsPureReadOperator(op.type)) {
		return false;
	}
	if (!allow_dependent && DependsOnSurroundingPlan(op.type)) {
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
		if (!IsCacheable(*child, allow_dependent)) {
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

//===--------------------------------------------------------------------===//
// Settings and logging
//===--------------------------------------------------------------------===//

static bool GetSetting(ClientContext &context, const string &name, Value &result) {
	return context.TryGetCurrentSetting(name, result) && !result.IsNull();
}

static bool BoolSetting(ClientContext &context, const string &name, bool fallback) {
	Value value;
	return GetSetting(context, name, value) ? value.GetValue<bool>() : fallback;
}

static idx_t CountSetting(ClientContext &context, const string &name, idx_t fallback) {
	Value value;
	if (!GetSetting(context, name, value)) {
		return fallback;
	}
	auto count = value.GetValue<int64_t>();
	return count < 0 ? 0 : NumericCast<idx_t>(count);
}

static void Log(ClientContext &context, const string &message) {
	if (BoolSetting(context, "waddle_log", false)) {
		Printer::Print("[waddle] " + message);
	}
}

//===--------------------------------------------------------------------===//
// Reuse: find the largest cached subtree and swap it in
//===--------------------------------------------------------------------===//

struct ReuseResult {
	bool replaced = false;
	string what;
	idx_t rows = 0;
};

static void TryReuse(unique_ptr<LogicalOperator> &slot, bool is_root, bool blocked, ResultCache &cache,
                     Optimizer &optimizer, LogicalOperator &root, ReuseResult &result) {
	if (result.replaced) {
		return;
	}
	auto &op = *slot;
	// The root of the query is always a candidate; below it we stay out of the
	// regions BlocksDescent names.
	if ((is_root || !blocked) && IsCacheable(op, is_root)) {
		string key;
		if (SerializePlan(op, key)) {
			auto cached = cache.Lookup(HashPlanKey(key), key);
			if (cached) {
				// The optimizers can leave an operator's types empty: DuckDB only
				// resolves them when it builds the physical plan. Without this the
				// check below compares against an empty list and every hit on such
				// a plan is thrown away.
				op.ResolveOperatorTypes();
			}
			auto bindings = op.GetColumnBindings();
			if (cached && cached->types == op.types && bindings.size() == op.types.size()) {
				auto table_index = optimizer.binder.GenerateTableIndex();
				result.replaced = true;
				result.what = is_root ? "the whole query" : op.GetName();
				result.rows = cached->data->Count();
				slot = make_uniq<LogicalCachedScan>(cached, table_index);
				// Everything above the swap still refers to columns the old
				// subtree produced; point those at the cached scan instead.
				if (!is_root) {
					RemapColumnBindings(root, bindings, table_index);
				}
				return;
			}
		}
	}
	auto blocked_below = blocked || BlocksDescent(op.type);
	for (auto &child : op.children) {
		TryReuse(child, false, blocked_below, cache, optimizer, root, result);
		if (result.replaced) {
			return;
		}
	}
}

//===--------------------------------------------------------------------===//
// Capture: pick what is worth remembering
//===--------------------------------------------------------------------===//

struct Candidate {
	optional_ptr<unique_ptr<LogicalOperator>> slot;
	string key;
	hash_t fingerprint = 0;
	idx_t operators = 0;
	idx_t rows = 0;
	idx_t work = 0;
};

//! Rows flowing through a subtree, as the optimizer estimates them: a rough
//! stand-in for how much work the subtree does.
static idx_t EstimatedWork(LogicalOperator &op, idx_t &operators) {
	operators++;
	idx_t work = op.has_estimated_cardinality ? op.estimated_cardinality : 0;
	for (auto &child : op.children) {
		work += EstimatedWork(*child, operators);
	}
	return work;
}

struct CandidateSearch {
	idx_t subtree_row_limit;
	idx_t min_operators;
	idx_t min_work;
	bool subtrees_enabled;
	//! The whole query, kept whatever it looks like: its result is the answer
	//! the next identical query needs, and it is the cheapest thing to store.
	Candidate root;
	//! Every subtree we could cache, in plan order.
	vector<Candidate> subtrees;
};

static void CollectCandidates(unique_ptr<LogicalOperator> &slot, bool is_root, bool blocked, CandidateSearch &search) {
	auto &op = *slot;
	auto consider = is_root || (search.subtrees_enabled && !blocked);
	if (consider && IsCacheable(op, is_root)) {
		idx_t operators = 0;
		auto work = EstimatedWork(op, operators);
		auto rows = op.has_estimated_cardinality ? op.estimated_cardinality : NumericLimits<idx_t>::Maximum();
		// The whole query is not judged by its estimate. The estimate of a
		// grouped aggregate is simply the size of its input, so a query that
		// returns a few thousand groups can look like hundreds of thousands of
		// rows; the capture operator enforces the row limit on the real result.
		auto within_limit = is_root || rows <= search.subtree_row_limit;
		auto worth_it = is_root || (operators >= search.min_operators && work >= search.min_work);
		if (within_limit && worth_it) {
			string key;
			if (SerializePlan(op, key)) {
				Candidate candidate;
				candidate.slot = &slot;
				candidate.fingerprint = HashPlanKey(key);
				candidate.key = std::move(key);
				candidate.operators = operators;
				candidate.rows = rows;
				candidate.work = work;
				if (is_root) {
					search.root = std::move(candidate);
				} else {
					search.subtrees.push_back(std::move(candidate));
				}
			}
		}
	}
	auto blocked_below = blocked || BlocksDescent(op.type);
	for (auto &child : op.children) {
		CollectCandidates(child, false, blocked_below, search);
	}
}

//! Wraps a plan, or part of one, so its rows are copied into the cache as the
//! query runs. Pass-through, so nothing above it needs rewriting.
static void WrapWithCapture(Candidate &candidate, shared_ptr<ResultCache> cache, idx_t max_bytes, idx_t max_rows) {
	auto &slot = *candidate.slot;
	auto capture = make_uniq<LogicalCaptureResult>(std::move(cache), candidate.fingerprint,
	                                               std::move(candidate.key), max_bytes, max_rows);
	capture->children.push_back(std::move(slot));
	capture->ResolveOperatorTypes();
	slot = std::move(capture);
}

//===--------------------------------------------------------------------===//
// The hook
//===--------------------------------------------------------------------===//

static void RecycleResults(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto &state = static_cast<RecyclerState &>(*input.info);
	auto &context = input.context;
	auto &cache = *state.cache;

	if (!BoolSetting(context, "waddle_cache", true)) {
		return;
	}
	cache.SetBudget(CountSetting(context, "waddle_cache_limit_mb", 512) * 1024 * 1024);
	auto root_row_limit = CountSetting(context, "waddle_cache_max_rows", 200000);
	auto subtree_row_limit = CountSetting(context, "waddle_subtree_max_rows", 200000);
	auto min_operators = CountSetting(context, "waddle_min_subtree_operators", 4);
	auto min_work = CountSetting(context, "waddle_min_subtree_work", 5000000);
	auto subtrees_enabled = BoolSetting(context, "waddle_subtree_reuse", true);

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
		cache.Clear();
		Log(context, StringUtil::Format("query %llu changes the database - cache emptied", query_number));
		return;
	}

	// 1. Can any part of this query be answered from something we already have?
	// `blocked` starts out true when subtree reuse is off, so the search looks
	// at the whole query only. Serializing every subtree of every plan is not
	// free, and it is wasted work if nothing below the root can be reused.
	ReuseResult reuse;
	TryReuse(slot.get(), true, !subtrees_enabled, cache, input.optimizer, *plan, reuse);
	if (reuse.replaced) {
		Log(context, StringUtil::Format("query %llu reuses %s from cache (%llu rows)", query_number, reuse.what,
		                                reuse.rows));
		return;
	}

	// 2. Otherwise run it as usual, but keep a copy of what is worth keeping.
	//    Every key is taken before anything is wrapped: a capture operator in
	//    the plan would change the bytes a later query has to match.
	CandidateSearch search;
	search.subtree_row_limit = subtree_row_limit;
	search.min_operators = min_operators;
	search.min_work = min_work;
	search.subtrees_enabled = subtrees_enabled;
	CollectCandidates(slot.get(), true, false, search);

	// Of the subtrees we could keep, take the one that does the most work and
	// that an earlier query already contained. Capturing a subtree costs real
	// time twice over - it has to come back often enough to pay for itself, and
	// it has to have been expensive to produce - so two conditions guard it:
	// the subtree must have appeared in an earlier query, and the optimizer
	// must think it processes at least `waddle_min_subtree_work` rows.
	// Materialising a cheap intermediate is measurably slower than recomputing
	// it: on a session whose shared part was a filtered scan feeding a much
	// bigger join, capturing it cost ~18% on the geometric mean.
	optional_ptr<Candidate> repeat;
	for (auto &candidate : search.subtrees) {
		auto sightings = state.RecordSighting(candidate.fingerprint);
		if (sightings >= 1 && (!repeat || candidate.work > repeat->work)) {
			repeat = &candidate;
		}
	}

	auto max_bytes = cache.MaxEntryBytes();
	string captured;
	// The subtree first: wrapping the root afterwards does not move it.
	if (repeat) {
		captured += StringUtil::Format(" a repeated %s (%llu operators, ~%llu rows)", repeat->slot->get()->GetName(),
		                               repeat->operators, repeat->rows);
		WrapWithCapture(*repeat, state.cache, max_bytes, subtree_row_limit);
	}
	if (search.root.slot) {
		captured += StringUtil::Format(" the whole query (~%llu rows)", search.root.rows);
		WrapWithCapture(search.root, state.cache, max_bytes, root_row_limit);
	}

	if (captured.empty()) {
		Log(context, StringUtil::Format("query %llu is new - nothing worth caching", query_number));
	} else {
		Log(context, StringUtil::Format("query %llu is new - capturing%s (%llu entries, %llu MB cached)", query_number,
		                                captured, cache.EntryCount(), cache.BytesUsed() / 1024 / 1024));
	}
}

OptimizerExtension CreateRecyclerOptimizer() {
	OptimizerExtension extension;
	extension.optimize_function = RecycleResults;
	extension.optimizer_info = make_shared_ptr<RecyclerState>();
	return extension;
}

} // namespace duckdb
