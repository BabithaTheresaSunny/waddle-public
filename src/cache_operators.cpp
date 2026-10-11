#include "cache_operators.hpp"

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Capture
//===--------------------------------------------------------------------===//

//! Per-thread state of the capture operator. The operator is not parallel, so
//! there is exactly one of these per query.
class CaptureState : public OperatorState {
public:
	unique_ptr<ColumnDataCollection> data;
	ColumnDataAppendState append_state;
	//! Set when the result outgrew the budget: we stop copying and give up on
	//! caching this plan, but the query itself runs on untouched.
	bool gave_up = false;
};

class PhysicalCaptureResult : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalCaptureResult(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                      shared_ptr<ResultCache> cache, hash_t fingerprint, string plan_key, idx_t max_bytes,
	                      idx_t max_rows)
	    : PhysicalOperator(physical_plan, TYPE, std::move(types), estimated_cardinality), cache(std::move(cache)),
	      fingerprint(fingerprint), plan_key(std::move(plan_key)), max_bytes(max_bytes), max_rows(max_rows) {
	}

	shared_ptr<ResultCache> cache;
	hash_t fingerprint;
	string plan_key;
	idx_t max_bytes;
	idx_t max_rows;

	string GetName() const override {
		return "CAPTURE_RESULT";
	}

	// ParallelOperator() stays false (the default): one thread runs this
	// operator, so rows are appended in a single, reproducible order. The price
	// is that the pipeline this operator sits in is not parallelised.
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override {
		auto state = make_uniq<CaptureState>();
		// Deliberately NOT the buffer manager: cached results outlive the query
		// that made them, and DuckDB tears the buffer manager down before
		// extension state, which would leave the cache pointing at freed
		// blocks. The cost is that these bytes sit outside memory_limit, which
		// is why the cache keeps its own budget.
		state->data = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
		state->data->InitializeAppend(state->append_state);
		return std::move(state);
	}

	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state_p) const override {
		auto &state = state_p.Cast<CaptureState>();
		if (!state.gave_up) {
			state.data->Append(state.append_state, input);
			if (state.data->AllocationSize() > max_bytes || state.data->Count() > max_rows) {
				state.gave_up = true;
				state.data.reset();
			}
		}
		// The rows still flow onwards, untouched.
		chunk.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	bool RequiresFinalExecute() const override {
		return true;
	}

	//! Called once, after the last row has passed through. Reaching this point
	//! means the plan below ran to completion, so the copy is a complete result.
	//! Called once, after the last row has passed through. Reaching this point
	//! means the plan below ran to completion, so the copy is a complete
	//! result.
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate,
	                                        OperatorState &state_p) const override {
		auto &state = state_p.Cast<CaptureState>();
		// An empty copy is never published. DuckDB sometimes runs the same
		// operator a second time within one query, and that run can see no
		// rows at all; publishing it would replace a complete result with
		// nothing. A subtree that genuinely returns no rows is cheap to
		// recompute, so skipping those costs nothing. The cache also keeps the
		// first result stored under a key, so a later run cannot overwrite it.
		if (!state.gave_up && state.data && state.data->Count() > 0) {
			shared_ptr<ColumnDataCollection> data = std::move(state.data);
			cache->Insert(fingerprint, plan_key, types, std::move(data));
		}
		chunk.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}
};

LogicalCaptureResult::LogicalCaptureResult(shared_ptr<ResultCache> cache_p, hash_t fingerprint_p, string plan_key_p,
                                           idx_t max_bytes_p, idx_t max_rows_p)
    : cache(std::move(cache_p)), fingerprint(fingerprint_p), plan_key(std::move(plan_key_p)), max_bytes(max_bytes_p),
      max_rows(max_rows_p) {
}

void LogicalCaptureResult::ResolveTypes() {
	types = children[0]->types;
}

vector<ColumnBinding> LogicalCaptureResult::GetColumnBindings() {
	// Pass-through: the rows above see exactly the columns the child produced,
	// so nothing above this operator needs rewriting.
	return children[0]->GetColumnBindings();
}

string LogicalCaptureResult::GetName() const {
	return "CAPTURE_RESULT";
}

string LogicalCaptureResult::GetExtensionName() const {
	return "waddle";
}

PhysicalOperator &LogicalCaptureResult::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	auto &child = planner.CreatePlan(*children[0]);
	auto &op = planner.Make<PhysicalCaptureResult>(types, estimated_cardinality, cache, fingerprint, plan_key, max_bytes,
	                                               max_rows);
	op.children.push_back(child);
	return op;
}

//===--------------------------------------------------------------------===//
// Cached scan
//===--------------------------------------------------------------------===//

class CachedScanState : public GlobalSourceState {
public:
	ColumnDataScanState scan_state;
};

class PhysicalCachedScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalCachedScan(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                   shared_ptr<CachedResult> result)
	    : PhysicalOperator(physical_plan, TYPE, std::move(types), estimated_cardinality), result(std::move(result)) {
	}

	//! Holding the shared_ptr keeps the rows alive for as long as this plan
	//! exists, even if the cache evicts the entry mid-query.
	shared_ptr<CachedResult> result;

	string GetName() const override {
		return "CACHED_SCAN";
	}

	bool IsSource() const override {
		return true;
	}

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		auto state = make_uniq<CachedScanState>();
		result->data->InitializeScan(state->scan_state);
		return std::move(state);
	}

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		auto &state = input.global_state.Cast<CachedScanState>();
		// Rows come back in the order they were produced the first time.
		if (!result->data->Scan(state.scan_state, chunk)) {
			return SourceResultType::FINISHED;
		}
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
};

LogicalCachedScan::LogicalCachedScan(shared_ptr<CachedResult> result_p, idx_t table_index_p)
    : result(std::move(result_p)), table_index(table_index_p) {
	types = result->types;
	SetEstimatedCardinality(result->data->Count());
}

void LogicalCachedScan::ResolveTypes() {
	types = result->types;
}

vector<ColumnBinding> LogicalCachedScan::GetColumnBindings() {
	return GenerateColumnBindings(table_index, types.size());
}

vector<idx_t> LogicalCachedScan::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalCachedScan::GetName() const {
	return "CACHED_SCAN";
}

string LogicalCachedScan::GetExtensionName() const {
	return "waddle";
}

PhysicalOperator &LogicalCachedScan::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	return planner.Make<PhysicalCachedScan>(types, estimated_cardinality, result);
}

} // namespace duckdb
