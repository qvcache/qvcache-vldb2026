// System headers
#include <cstddef>
#include <omp.h>
#include <boost/program_options.hpp>
#include <atomic>
#include <iomanip>
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <cmath>
#include <limits>
#include <utility>

// Backend header
#include "greator_backend.h"

// DiskANN headers
#include "diskann/distance.h"
#include "greator/utils.h"

namespace po = boost::program_options;

// Metrics structures (simplified, no QVCache-specific metrics)
struct BackendMetrics {
    size_t total_queries;
    uint32_t threads;
    double avg_latency_ms;
    double qps;
    double qps_per_thread;
    double p50;
    double p90;
    double p95;
    double p99;
};

struct RecallAllMetrics {
    double recall_all;
    uint32_t K;
    size_t low_recall_queries;
    size_t very_low_recall_queries;
};

template <typename T, typename TagT = uint32_t>
RecallAllMetrics calculate_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    std::vector<double> recall_by_query;
    const TagT INVALID_ID = std::numeric_limits<TagT>::max();
    
    for (int32_t i = 0; i < query_num; i++) {
        std::set<uint32_t> groundtruth_closest_neighbors;
        std::set<uint32_t> calculated_closest_neighbors;
        for (int32_t j = 0; j < K; j++) {
            groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
        }
        // Filter out invalid IDs (padded results)
        for (int32_t j = 0; j < K; j++) {
            TagT tag = *(query_result_tags.data() + i * K + j);
            if (tag != INVALID_ID) {
                calculated_closest_neighbors.insert(tag);
            }
        }
        uint32_t matching_neighbors = 0;
        for (uint32_t x : calculated_closest_neighbors) {
            // Backend returns tags directly from index (0-indexed), which should match groundtruth
            // Groundtruth is typically 0-indexed, so we use x directly (not x-1)
            if (groundtruth_closest_neighbors.count(x)) {
                matching_neighbors++;
            }
        }
        double recall = matching_neighbors / (double)K;
        recall_by_query.push_back(recall);
        total_recall += recall;
    }
    double average_recall = total_recall / (query_num);
    
    // Count queries with low recall
    size_t low_recall_count = 0;
    size_t very_low_recall_count = 0;
    for (double r : recall_by_query) {
        if (r < 0.5) low_recall_count++;
        if (r < 0.1) very_low_recall_count++;
    }
    
    RecallAllMetrics metrics;
    metrics.recall_all = average_recall;
    metrics.K = K;
    metrics.low_recall_queries = low_recall_count;
    metrics.very_low_recall_queries = very_low_recall_count;
    
    return metrics;
}

template <typename T, typename TagT = uint32_t>
void log_split_metrics(int split_idx, const BackendMetrics& backend_metrics, const RecallAllMetrics& recall_all_metrics) {
    // Log combined metrics in single JSON line
    spdlog::info("{{\"event\": \"split_metrics\", "
              "\"split_idx\": {}, "
              "\"total_queries\": {}, "
              "\"threads\": {}, "
              "\"avg_latency_ms\": {}, "
              "\"qps\": {}, "
              "\"qps_per_thread\": {}, "
              "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
              "\"recall_all\": {}, "
              "\"K\": {}, "
              "\"low_recall_queries\": {}, "
              "\"very_low_recall_queries\": {}}}",
              split_idx,
              backend_metrics.total_queries,
              backend_metrics.threads, backend_metrics.avg_latency_ms,
              backend_metrics.qps, backend_metrics.qps_per_thread,
              backend_metrics.p50, backend_metrics.p90, backend_metrics.p95, backend_metrics.p99,
              recall_all_metrics.recall_all, recall_all_metrics.K,
              recall_all_metrics.low_recall_queries, recall_all_metrics.very_low_recall_queries);
}

template <typename T, typename TagT = uint32_t>
std::pair<BackendMetrics, std::vector<TagT>> backend_search(
    qvcache::BackendInterface<T, TagT>& backend,
    const T* query, size_t query_num, uint32_t query_aligned_dim,
    uint32_t K, uint32_t search_threads
) {
    std::vector<TagT> query_result_tags(K * query_num);
    std::vector<float> query_result_dists(K * query_num);
    greator::QueryStats* stats = new greator::QueryStats[query_num];
    std::vector<double> latencies_ms(query_num, 0.0);
    
    auto global_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        backend.search(
            query + i * query_aligned_dim,
            K,
            query_result_tags.data() + i * K,
            query_result_dists.data() + i * K,
            nullptr,
            stats + i
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    auto global_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(global_end - global_start).count();
    double total_time_sec = total_time_ms / 1000.0;
    double avg_latency_ms = 0.0;
    for (double latency : latencies_ms) avg_latency_ms += latency;
    avg_latency_ms /= query_num;
    double qps = static_cast<double>(query_num) / total_time_sec;
    double qps_per_thread = qps / static_cast<double>(search_threads);
    
    std::vector<double> sorted_latencies = latencies_ms;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    auto get_percentile = [&](double p) {
        size_t idx = static_cast<size_t>(std::ceil(p * query_num)) - 1;
        if (idx >= query_num) idx = query_num - 1;
        return sorted_latencies[idx];
    };
    double p50 = get_percentile(0.50);
    double p90 = get_percentile(0.90);
    double p95 = get_percentile(0.95);
    double p99 = get_percentile(0.99);
    
    BackendMetrics metrics;
    metrics.total_queries = query_num;
    metrics.threads = search_threads;
    metrics.avg_latency_ms = avg_latency_ms;
    metrics.qps = qps;
    metrics.qps_per_thread = qps_per_thread;
    metrics.p50 = p50;
    metrics.p90 = p90;
    metrics.p95 = p95;
    metrics.p99 = p99;
    
    delete[] stats;
    return {metrics, query_result_tags};
}

// Main experiment logic for backend-only cache pressure benchmark
template <typename T = float, typename TagT = uint32_t>
void experiment_benchmark(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& disk_index_prefix,
    uint32_t R, uint32_t disk_L, uint32_t K,
    uint32_t B, uint32_t M,
    uint32_t build_threads,
    uint32_t search_threads,
    int disk_index_already_built,
    uint32_t beamwidth,
    int n_splits,
    int n_split_repeat,
    int n_round,
    diskann::Metric metric
) {
    // Create backend
    std::unique_ptr<qvcache::BackendInterface<T, TagT>> backend = std::make_unique<qvcache::GreatorBackend<T>>(
        data_path, disk_index_prefix, R, disk_L, B, M, build_threads, disk_index_already_built, beamwidth);

    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    greator::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    greator::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);
    
    // Query file structure: all copies of split 0, then all copies of split 1, etc.
    // Each split has n_split_repeat copies, and each copy has queries_per_original_split queries
    size_t queries_per_original_split = query_num / (n_splits * n_split_repeat);
    
    // Calculate copies per round: divide n_split_repeat copies across n_round rounds
    int copies_per_round = n_split_repeat / n_round;
    if (copies_per_round * n_round != n_split_repeat) {
        std::cerr << "Error: n_split_repeat (" << n_split_repeat << ") must be divisible by n_round (" << n_round << ")" << std::endl;
        exit(1);
    }
    
    // Repeat the split processing loop n_round times
    for (int round = 0; round < n_round; ++round) {
        spdlog::info("{{\"event\": \"round_start\", \"round\": {}}}", round);
        
        // Calculate copy range for this round
        int copy_start = round * copies_per_round;
        int copy_end = (round + 1) * copies_per_round;
        
        // Process splits one by one sequentially
        for (int split_idx = 0; split_idx < n_splits; ++split_idx) {
            spdlog::info("{{\"event\": \"split_start\", \"split_idx\": {}, \"round\": {}}}", split_idx, round);
            
            // Process copies for this round (subset of n_split_repeat copies)
            for (int copy_idx = copy_start; copy_idx < copy_end; ++copy_idx) {
                // Calculate query range for this specific copy of this split
                // Structure: split 0 (all copies), split 1 (all copies), ...
                // For split i, copy j: offset = i * (n_split_repeat * queries_per_original_split) + j * queries_per_original_split
                size_t split_offset = split_idx * n_split_repeat * queries_per_original_split;
                size_t copy_offset = copy_idx * queries_per_original_split;
                size_t query_start = split_offset + copy_offset;
                size_t query_end = std::min(query_start + queries_per_original_split, query_num);
                
                if (query_start >= query_end) continue;
                
                size_t this_split_size = query_end - query_start;
                
                // Perform search using backend
                auto [backend_metrics, query_result_tags] = backend_search<T, TagT>(
                    *backend,
                    query + query_start * query_aligned_dim,
                    this_split_size,
                    query_aligned_dim,
                    K,
                    search_threads
                );
                
                // Calculate groundtruth offset (same structure as queries)
                size_t gt_start = split_offset + copy_offset;
                RecallAllMetrics recall_all = calculate_recall<T, TagT>(
                    K, groundtruth_ids + gt_start * groundtruth_dim, 
                    query_result_tags, this_split_size, groundtruth_dim);
                
                log_split_metrics<T, TagT>(split_idx, backend_metrics, recall_all);
            }
            
            spdlog::info("{{\"event\": \"split_end\", \"split_idx\": {}, \"round\": {}}}", split_idx, round);
        }
        
        spdlog::info("{{\"event\": \"round_end\", \"round\": {}}}", round);
    }
    
    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path, disk_index_prefix;
    uint32_t R, disk_L, K, B, M;
    uint32_t build_threads, search_threads, beamwidth;
    int disk_index_already_built;
    int n_splits;
    int n_split_repeat;
    int n_round = 1;
    uint32_t sector_len = 4096;
    std::string metric_str = "l2";
    po::options_description desc;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "Prefix to index")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("disk_L", po::value<uint32_t>(&disk_L)->required(), "Value of disk L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1), "Disk index already built (0/1)")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("n_splits", po::value<int>(&n_splits)->required(), "Number of splits for queries")
            ("n_split_repeat", po::value<int>(&n_split_repeat)->required(), "Number of repeats per split pattern")
            ("n_round", po::value<int>(&n_round)->default_value(1), "Number of rounds to repeat the entire experiment")
            ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes")
            ("metric", po::value<std::string>(&metric_str)->default_value("l2"), "Distance metric: l2, cosine, or inner_product");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    // Parse metric string to enum
    diskann::Metric metric;
    if (metric_str == "l2" || metric_str == "L2") {
        metric = diskann::Metric::L2;
    } else if (metric_str == "cosine" || metric_str == "COSINE") {
        metric = diskann::Metric::COSINE;
    } else if (metric_str == "inner_product" || metric_str == "INNER_PRODUCT" || metric_str == "innerproduct") {
        metric = diskann::Metric::INNER_PRODUCT;
    } else {
        std::cerr << "Unsupported metric: " << metric_str << ". Supported metrics: l2, cosine, inner_product" << std::endl;
        return -1;
    }
    
    set_sector_len(sector_len);
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v");
    logger->info("{{\n"
        "  \"event\": \"params\",\n"
        "  \"data_type\": \"{}\",\n"
        "  \"data_path\": \"{}\",\n"
        "  \"query_path\": \"{}\",\n"
        "  \"groundtruth_path\": \"{}\",\n"
        "  \"disk_index_prefix\": \"{}\",\n"
        "  \"R\": {},\n"
        "  \"disk_L\": {},\n"
        "  \"K\": {},\n"
        "  \"B\": {},\n"
        "  \"M\": {},\n"
        "  \"build_threads\": {},\n"
        "  \"search_threads\": {},\n"
        "  \"disk_index_already_built\": {},\n"
        "  \"beamwidth\": {},\n"
        "  \"n_splits\": {},\n"
        "  \"n_split_repeat\": {},\n"
        "  \"n_round\": {},\n"
        "  \"sector_len\": {},\n"
        "  \"metric\": \"{}\"\n"
        "}}",
        data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, disk_L, K, B, M, build_threads, search_threads, disk_index_already_built, beamwidth, n_splits, n_split_repeat, n_round, sector_len, metric_str);
    if (data_type == "float") {
        experiment_benchmark<float>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, disk_L, K, B, M, build_threads, search_threads, disk_index_already_built, beamwidth, n_splits, n_split_repeat, n_round, metric);
    } else if (data_type == "int8") {
        experiment_benchmark<int8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, disk_L, K, B, M, build_threads, search_threads, disk_index_already_built, beamwidth, n_splits, n_split_repeat, n_round, metric);
    } else if (data_type == "uint8") {
        experiment_benchmark<uint8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, disk_L, K, B, M, build_threads, search_threads, disk_index_already_built, beamwidth, n_splits, n_split_repeat, n_round, metric);
    } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
    }
    return 0;
}

