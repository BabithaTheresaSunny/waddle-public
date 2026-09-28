#include "plan_key.hpp"

#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/types/hash.hpp"

namespace duckdb {

bool SerializePlan(const LogicalOperator &op, string &result) {
	try {
		MemoryStream stream(Allocator::DefaultAllocator());
		BinarySerializer serializer(stream);
		serializer.Begin();
		op.Serialize(serializer);
		serializer.End();
		result = string(const_char_ptr_cast(stream.GetData()), stream.GetPosition());
		return true;
	} catch (std::exception &) {
		// Not every operator or function can be serialized. Those queries are
		// simply not cached.
		return false;
	}
}

hash_t HashPlanKey(const string &key) {
	return Hash(key.c_str(), key.size());
}

static idx_t Visit(const LogicalOperator &op, vector<SubtreeInfo> &out) {
	idx_t size = 1;
	for (auto &child : op.children) {
		size += Visit(*child, out);
	}
	string key;
	if (SerializePlan(op, key)) {
		out.push_back(SubtreeInfo {HashPlanKey(key), &op, size});
	}
	return size;
}

vector<SubtreeInfo> FingerprintSubtrees(const LogicalOperator &root) {
	vector<SubtreeInfo> result;
	Visit(root, result);
	return result;
}

} // namespace duckdb
