#define DUCKDB_EXTENSION_MAIN

#include "waddle_extension.hpp"
#include "recycler_optimizer.hpp"
#include "duckdb.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("waddle_cache", "Reuse the result of a query that has run before",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption("waddle_cache_limit_mb", "Memory the result cache may use, in MB", LogicalType::BIGINT,
	                          Value::BIGINT(512));
	config.AddExtensionOption("waddle_cache_max_rows", "Largest result (in rows) worth caching", LogicalType::BIGINT,
	                          Value::BIGINT(200000));
	config.AddExtensionOption("waddle_log", "Print what the recycler does with each query", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(false));
	OptimizerExtension::Register(config, CreateRecyclerOptimizer());
}

void WaddleExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string WaddleExtension::Name() {
	return "waddle";
}
std::string WaddleExtension::Version() const {
#ifdef EXT_VERSION_WADDLE
	return EXT_VERSION_WADDLE;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(waddle, loader) {
	duckdb::LoadInternal(loader);
}
}
