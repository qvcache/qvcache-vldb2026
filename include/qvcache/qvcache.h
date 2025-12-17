// DiskANN (FreshVamana) headers
#include "diskann/utils.h"
#include "diskann/index_factory.h"

// QVCache headers
#include "qvcache/insert_thread_pool.h"
#include "qvcache/pca_utils.h"
#include "qvcache/lru_cache.h"
#include "qvcache/backend_interface.h" 

// System headers
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <type_traits>
#include <memory>
#include <cstring>
#include <atomic>
#include <future>
#include <vector>
#include <algorithm>
#include <deque>
#include <limits>

namespace qvcache {

    template <typename T, typename TagT = uint32_t>
    class QVCache {
        
        private:
            // Backend vector database
            std::unique_ptr<BackendInterface<T, TagT>> backend;
            
            // LRU-managed memory indices: n memory indices managed by LRU eviction
            std::vector<std::unique_ptr<diskann::AbstractIndex>> memory_indices;
            std::atomic<size_t> active_insert_index_id; // Index ID for new insertions

            // --- LRU Cache for managing mini-index access patterns ---
            std::unique_ptr<LRUCache<size_t>> lru_cache; // Tracks mini-index usage

            // --- Index parameters ---
            std::string data_path;
            std::string pca_prefix;
            size_t dim, aligned_dim;
            size_t num_points;
            size_t memory_index_max_points_per_index; // Capacity per index
            size_t number_of_mini_indexes; // Number of mini indexes
            uint32_t search_threads;
            bool use_reconstructed_vectors;
            std::unique_ptr<qvcache::InsertThreadPool<T, TagT>> insert_pool;
            std::unordered_map<uint32_t, double> theta_map;
            std::mutex theta_map_mutex;
            double p, deviation_factor;
            uint32_t memory_L; 
            uint32_t beamwidth;
            bool use_regional_theta = true;

            uint32_t n_async_insert_threads = 4;
            bool lazy_theta_updates = true;
            bool search_mini_indexes_in_parallel = false; // Control parallel vs sequential search
            size_t max_search_threads = 32; // Maximum threads for parallel search (should be > query processing threads)
            diskann::Metric metric = diskann::L2; // Distance metric (default: L2)

            // --- LRU eviction state ---
            std::atomic<bool> eviction_in_progress{false};
            std::future<void> eviction_future;
            std::mutex eviction_mutex;

            // --- PCA utilities ---
            std::unique_ptr<PCAUtils<T>> pca_utils;

        public:
            // Configuration option to choose search strategy
            enum class SearchStrategy {
                SEQUENTIAL_LRU_STOP_FIRST_HIT,  // Problematic: Stop at first hit in LRU order (causes recall drops)
                SEQUENTIAL_LRU_ADAPTIVE,  // Adaptive: Monitor hit ratio, switch to SEQUENTIAL_ALL when low
                SEQUENTIAL_ALL,           // Search all indices sequentially, pick best result
                PARALLEL                  // Parallel search (existing implementation)
            };

        private:
            // Search strategy for the public search method
            SearchStrategy search_strategy = SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT;
            
            // Hit ratio monitoring for adaptive strategy
            std::deque<bool> hit_history;  // Sliding window of hit/miss results
            std::mutex hit_history_mutex;
            size_t hit_ratio_window_size = 100;  // Default window size
            double hit_ratio_threshold = 0.90;   // Default threshold (90%)
            bool use_adaptive_strategy = false;  // Whether to use adaptive behavior

            // Helper function to create a memory index with given max points
            std::unique_ptr<diskann::AbstractIndex> create_memory_index(size_t max_points) {
                diskann::IndexWriteParameters memory_index_write_params = diskann::IndexWriteParametersBuilder(memory_L, aligned_dim)
                                                                    .with_alpha(1.2f) // Default alpha
                                                                    .with_num_threads(4) // Default threads
                                                                    .build();

                diskann::IndexSearchParams memory_index_search_params = diskann::IndexSearchParams(memory_L, search_threads);

                diskann::IndexConfig memory_index_config = diskann::IndexConfigBuilder()
                                                            .with_metric(metric)
                                                            .with_dimension(dim)
                                                            .with_max_points(max_points)
                                                            .is_dynamic_index(true)
                                                            .with_index_write_params(memory_index_write_params)
                                                            .with_index_search_params(memory_index_search_params)
                                                            .with_data_type(diskann_type_to_name<T>())
                                                            .with_tag_type(diskann_type_to_name<TagT>())
                                                            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                                            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                                            .is_enable_tags(true)
                                                            .is_filtered(false)
                                                            .with_num_frozen_pts(0)
                                                            .is_concurrent_consolidate(true)
                                                            .build();
                
                diskann::IndexFactory memory_index_factory = diskann::IndexFactory(memory_index_config);
                auto index = memory_index_factory.create_instance();
                index->set_start_points_at_random(static_cast<T>(0));
                return index;
            }

            void memory_index_insert_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted) {
                // Use backend to fetch vectors by IDs
                std::vector<std::vector<T>> fetched_vectors = backend->fetch_vectors_by_ids(to_be_inserted);
                
                // Allocate aligned memory for vectors and copy from fetched vectors
                std::vector<T*> vectors;
                vectors.reserve(to_be_inserted.size());
                for (size_t i = 0; i < fetched_vectors.size(); ++i) {
                    T* vector = nullptr;
                    diskann::alloc_aligned((void**)&vector, aligned_dim * sizeof(T), 8 * sizeof(T));
                    // Copy the fetched vector data
                    std::memcpy(vector, fetched_vectors[i].data(), dim * sizeof(T));
                    // Zero out padding if needed
                    if (aligned_dim > dim) {
                        std::memset(vector + dim, 0, (aligned_dim - dim) * sizeof(T));
                    }
                    vectors.push_back(vector);
                }
                
                size_t successful_inserts = 0;
                // Insert the new vectors into the provided index (which should be the current active one)
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    int ret = index->insert_point(vectors[i], 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                
                // Check if we need to trigger LRU eviction AFTER insertion
                // Only trigger if the currently active index is full
                size_t current_active_id = active_insert_index_id.load();
                if (index.get() == memory_indices[current_active_id].get() && index->get_number_of_active_vectors() >= memory_index_max_points_per_index) {
                    // Always evict when the currently active index is full
                    if (!eviction_in_progress.load()) {
                        // Start LRU eviction
                        trigger_eviction();
                    }
                }
                
                for (auto v : vectors) {
                    diskann::aligned_free(v);
                }
            }

            void memory_index_insert_reconstructed_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted) {
                // Use disk_backend to fetch vectors
                std::vector<std::vector<T>> reconstructed_vectors = this->backend->fetch_vectors_by_ids(to_be_inserted);
                size_t successful_inserts = 0;
                
                // Insert the reconstructed vectors into the provided index
                for (size_t i = 0; i < to_be_inserted.size(); i++) {
                    const auto& reconstructed_vec = reconstructed_vectors[i];
                    int ret = index->insert_point(reconstructed_vec.data(), 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                
                // Check if we need to trigger LRU eviction AFTER insertion
                // Only trigger if the currently active index is full
                size_t current_active_id = active_insert_index_id.load();
                if (index.get() == memory_indices[current_active_id].get() && index->get_number_of_active_vectors() >= memory_index_max_points_per_index) {
                    // Always evict when the currently active index is full
                    if (!eviction_in_progress.load()) {
                        trigger_eviction();
                    }
                }
            }

            // LRU eviction: evict the least recently used index and replace with fresh one
            void trigger_eviction() {
                std::lock_guard<std::mutex> lock(eviction_mutex);
                
                if (eviction_in_progress.load()) {
                    return; // Already in progress
                }
                
                eviction_in_progress.store(true);
                
                // Start background eviction
                eviction_future = std::async(std::launch::async, [this]() {
                    this->perform_eviction();
                });
            }

            void perform_eviction() {
                // Debug: Show current LRU order before eviction
                std::vector<size_t> current_lru_order = lru_cache->get_all_tags();
                std::cout << "[LRU Eviction] Current LRU order before eviction: ";
                for (size_t idx : current_lru_order) {
                    std::cout << idx << " ";
                }
                std::cout << std::endl;
                
                // Get the least recently used index ID from LRU cache
                std::vector<size_t> lru_tags = lru_cache->get_lru_tags(1);
                if (lru_tags.empty()) {
                    // Fallback to cycling if LRU cache is empty
                    size_t current_active_id = active_insert_index_id.load();
                    size_t next_active_id = (current_active_id + 1) % number_of_mini_indexes;
                    std::cout << "[LRU Eviction] Fallback: Evicting index " << next_active_id << std::endl;
                    memory_indices[next_active_id] = create_memory_index(memory_index_max_points_per_index);
                    active_insert_index_id.store(next_active_id);
                } else {
                    size_t lru_index_id = lru_tags[0];
                    
                    // Check if this is the only index in the cache
                    if (current_lru_order.size() == 1) {
                        // If only one index remains, don't evict it - this prevents infinite eviction loop
                        std::cout << "[LRU Eviction] Only one index remaining (" << lru_index_id << "), skipping eviction to prevent infinite loop" << std::endl;
                        eviction_in_progress.store(false);
                        return;
                    }
                    
                    std::cout << "[LRU Eviction] Evicting least recently used index: " << lru_index_id << std::endl;
                    // STEP 1: Replace the LRU index with a fresh empty one
                    memory_indices[lru_index_id] = create_memory_index(memory_index_max_points_per_index);
                    // STEP 2: Switch active insert index to the fresh empty index
                    active_insert_index_id.store(lru_index_id);
                    // STEP 3: Remove the evicted index from LRU cache and add the new active index
                    // Note: We don't need to explicitly evict since we're replacing the same index ID
                    lru_cache->access(lru_index_id); // This will move the index to front (most recently used)
                }
                
                // Debug: Show new LRU order after eviction
                std::vector<size_t> new_lru_order = lru_cache->get_all_tags();
                std::cout << "[LRU Eviction] New LRU order after eviction: ";
                for (size_t idx : new_lru_order) {
                    std::cout << idx << " ";
                }
                std::cout << std::endl;
                std::cout << "[LRU Eviction] LRU cache size: " << lru_cache->size() << "/" << lru_cache->max_capacity() << std::endl;
                
                // Ensure all indices are present in the LRU cache
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    if (!lru_cache->contains(i)) {
                        std::cout << "[LRU Eviction] Warning: Index " << i << " missing from LRU cache, adding it" << std::endl;
                        lru_cache->access(i);
                    }
                }
                
                // Eviction complete
                eviction_in_progress.store(false);
            }

            bool isHit(const T* query_ptr, uint32_t K, const float* distances) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                
                if (use_regional_theta) {
                    return pca_utils->isHit(query_ptr, K, distances, this->get_number_of_vectors_in_memory_index(), deviation_factor);
                } else {
                    if (this->get_number_of_vectors_in_memory_index() < K){
                        return false;
                    }
                    double threshold = theta_map[K];
                    // Handle uninitialized thresholds
                    bool is_uninitialized = (metric == diskann::COSINE)
                        ? (threshold == -std::numeric_limits<double>::infinity())
                        : (threshold >= std::numeric_limits<double>::max() * 0.5);
                    if (is_uninitialized) {
                        return false; // Always miss until threshold is initialized
                    }
                    // Use multiplicative tolerance for both L2 and cosine
                    double cache_distance = static_cast<double>(distances[K - 1]);
                    double tolerance_threshold = (1.0 + deviation_factor) * threshold;
                    
                    if (cache_distance > tolerance_threshold) {
                        return false;
                    }
                    return true;
                }
            }

            void update_theta(const T* query_ptr, uint32_t K, float query_distance) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                if (use_regional_theta) {
                    pca_utils->update_theta(query_ptr, K, query_distance, p);
                } else {
                    double current_theta = theta_map[K];
                    // Handle initialization: if current_theta is uninitialized, replace it completely
                    bool is_uninitialized = (metric == diskann::COSINE)
                        ? (current_theta == -std::numeric_limits<double>::infinity())
                        : (current_theta >= std::numeric_limits<double>::max() * 0.5);
                    if (is_uninitialized) {
                        theta_map[K] = static_cast<double>(query_distance);
                    } else {
                        theta_map[K] = p * static_cast<double>(query_distance) + (1 - p) * current_theta;
                    }
                }
            }


            // Helper function to search a single memory index
            bool search_single_index(size_t index_id, const T* query_ptr, uint32_t K,  
                                   uint32_t* query_result_tags_ptr, std::vector<T*>& res, 
                                   float* query_result_dists_ptr) {
                if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                    memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                    bool is_hit = this->isHit(query_ptr, K, query_result_dists_ptr);
                    
                    // Only update LRU cache if this search resulted in a hit
                    if (is_hit) {
                        lru_cache->access(index_id);
                    }
                    
                    return is_hit;
                }
                return false;
            }

            // Parallel search across all memory indices using thread pool
            bool parallel_search_memory_indices(const T* query_ptr, uint32_t K, 
                                              uint32_t* query_result_tags_ptr, std::vector<T*>& res,
                                              float* query_result_dists_ptr) {
                if (number_of_mini_indexes == 1) {
                    // Single index case - no need for parallelization
                    return search_single_index(0, query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr);
                }

                // Prepare results for each index
                std::vector<std::vector<uint32_t>> all_tags(number_of_mini_indexes);
                std::vector<std::vector<float>> all_dists(number_of_mini_indexes);
                std::vector<std::vector<T*>> all_res(number_of_mini_indexes);
                std::vector<bool> hit_results(number_of_mini_indexes, false);
                std::vector<std::mutex> result_mutexes(number_of_mini_indexes);
                std::vector<std::future<void>> futures;

                // Lambda function for parallel search
                auto search_worker = [&](size_t index_id) {
                    if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                        // Allocate temporary storage for this thread
                        std::vector<uint32_t> temp_tags(K);
                        std::vector<float> temp_dists(K);
                        std::vector<T*> temp_res;

                        // Search in this index
                        memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, temp_tags.data(), temp_dists.data(), temp_res);
                        
                        // Check if this is a hit
                        bool is_hit = this->isHit(query_ptr, K, temp_dists.data());
                        
                        // Only update LRU cache if this search resulted in a hit
                        if (is_hit) {
                            lru_cache->access(index_id);
                        }
                        
                        // Store results atomically
                        {
                            std::lock_guard<std::mutex> lock(result_mutexes[index_id]);
                            all_tags[index_id] = std::move(temp_tags);
                            all_dists[index_id] = std::move(temp_dists);
                            all_res[index_id] = std::move(temp_res);
                            hit_results[index_id] = is_hit;
                        }
                    }
                };

                // Submit tasks to thread pool
                size_t worker_count = std::min(max_search_threads, number_of_mini_indexes);
                if (worker_count == 0) {
                    return false;
                }

                std::atomic<size_t> next_index{0};
                auto worker_loop = [&]() {
                    while (true) {
                        size_t index_id = next_index.fetch_add(1, std::memory_order_relaxed);
                        if (index_id >= number_of_mini_indexes) {
                            break;
                        }
                        search_worker(index_id);
                    }
                };

                for (size_t i = 0; i < worker_count; ++i) {
                    futures.emplace_back(std::async(std::launch::async, worker_loop));
                }

                // Wait for all tasks to complete
                for (auto& future : futures) {
                    future.get();
                }

                // Find the first hit and use its results
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    if (hit_results[i]) {
                        // Copy the winning results to the output
                        std::copy(all_tags[i].begin(), all_tags[i].end(), query_result_tags_ptr);
                        std::copy(all_dists[i].begin(), all_dists[i].end(), query_result_dists_ptr);
                        res = std::move(all_res[i]);
                        return true;
                    }
                }

                return false; // No hits found
            }



            // Problematic search strategy: stop at first hit in LRU order (causes recall drops)
            bool search_sequential_lru_stop_first_hit(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                // Get current active insert index ID (atomic read)
                size_t current_active_id = active_insert_index_id.load();
                
                // Sequential search in LRU order (most recently used first)
                std::vector<size_t> lru_order = lru_cache->get_all_tags();
                
                for (size_t index_id : lru_order) {
                    if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                        memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                        bool is_hit = this->isHit(query_ptr, K, query_result_dists_ptr);
                        
                        // Only update LRU cache if this search resulted in a hit
                        if (is_hit) {
                            lru_cache->access(index_id);
                            return true; // Found a hit, stop immediately (this was the problem!)
                        }
                    }
                }
                
                // No hit found in memory indices, search disk using the backend interface
                this->backend->search(query_ptr, (uint64_t)K, query_result_tags_ptr, query_result_dists_ptr, nullptr, backend_stats);
                std::vector<uint32_t> tags_to_insert(query_result_tags_ptr, query_result_tags_ptr + K);
                
                if (lazy_theta_updates) {
                    // Copy query pointer for async insertion and theta update
                    T* query_copy = nullptr;
                    diskann::alloc_aligned((void**)&query_copy, this->aligned_dim * sizeof(T), 8 * sizeof(T));
                    std::memcpy(query_copy, query_ptr, this->aligned_dim * sizeof(T));
                    
                    // Submit to insert pool with current active index
                    insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1], query_copy);
                } else {
                    // Immediate theta update in main thread
                    update_theta(query_ptr, K, query_result_dists_ptr[K - 1]);
                    insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1]);
                }
                
                for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                return false; // Return false if the query is missed in the memory index
            }

            // Adaptive search strategy: monitor hit ratio and switch to SEQUENTIAL_ALL when low
            bool search_adaptive_hit_ratio(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                // Check if we should use adaptive strategy (hit ratio is low)
                bool use_adaptive = should_use_adaptive_strategy();
                
                bool was_hit;
                
                if (use_adaptive) {
                    // Use SEQUENTIAL_ALL strategy when hit ratio is low
                    was_hit = search_sequential_all_impl(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                } else {
                    // Use SEQUENTIAL_LRU_STOP_FIRST_HIT strategy when hit ratio is good
                    was_hit = search_sequential_lru_stop_first_hit(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                }
                
                // Update hit history for monitoring
                update_hit_history(was_hit);
                
                return was_hit;
            }
            
            // Helper method for SEQUENTIAL_ALL implementation
            bool search_sequential_all_impl(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                // Get current active insert index ID (atomic read)
                size_t current_active_id = active_insert_index_id.load();
                
                // Sequential search in LRU order (most recently used first)
                std::vector<size_t> lru_order = lru_cache->get_all_tags();
                
                // Search indices in LRU order (most recently used first)
                // But don't stop at first hit - search all indices to find the best result
                bool found_hit = false;
                float best_distance = std::numeric_limits<float>::max();
                size_t best_index_id = 0;
                std::vector<uint32_t> best_tags;
                std::vector<float> best_dists;
                std::vector<T*> best_res;
                
                for (size_t index_id : lru_order) {
                    if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                        // Allocate temporary storage for this search
                        std::vector<uint32_t> temp_tags(K);
                        std::vector<float> temp_dists(K);
                        std::vector<T*> temp_res;
                        
                        memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, temp_tags.data(), temp_dists.data(), temp_res);
                        bool is_hit = this->isHit(query_ptr, K, temp_dists.data());
                        
                        if (is_hit) {
                            found_hit = true;
                            // Check if this result is better than the current best
                            // Use the K-th distance as a quality metric
                            if (temp_dists[K-1] < best_distance) {
                                best_distance = temp_dists[K-1];
                                best_index_id = index_id;
                                best_tags = std::move(temp_tags);
                                best_dists = std::move(temp_dists);
                                best_res = std::move(temp_res);
                            }
                        }
                    }
                }
                
                if (found_hit) {
                    // Use the best result found
                    std::copy(best_tags.begin(), best_tags.end(), query_result_tags_ptr);
                    std::copy(best_dists.begin(), best_dists.end(), query_result_dists_ptr);
                    res = std::move(best_res);
                    
                    // Update LRU cache for the index that provided the best result
                    lru_cache->access(best_index_id);
                    
                    return true;
                }
                
                // No hit found in memory indices, search disk using the backend interface
                this->backend->search(query_ptr, (uint64_t)K, query_result_tags_ptr, query_result_dists_ptr, nullptr, backend_stats);
                std::vector<uint32_t> tags_to_insert(query_result_tags_ptr, query_result_tags_ptr + K);
                
                if (lazy_theta_updates) {
                    // Copy query pointer for async insertion and theta update
                    T* query_copy = nullptr;
                    diskann::alloc_aligned((void**)&query_copy, this->aligned_dim * sizeof(T), 8 * sizeof(T));
                    std::memcpy(query_copy, query_ptr, this->aligned_dim * sizeof(T));
                    
                    // Submit to insert pool with current active index
                    insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1], query_copy);
                } else {
                    // Immediate theta update in main thread
                    update_theta(query_ptr, K, query_result_dists_ptr[K - 1]);
                    insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1]);
                }
                
                for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                return false; // Return false if the query is missed in the memory index
            }

            void load_sampled_data(const std::string& data_path, T*& sampled_data, size_t& sampled_num_points, size_t aligned_dim, size_t total_num_points, size_t sample_rate = 1000) {
                // Calculate the number of sampled points
                sampled_num_points = total_num_points / sample_rate;
                sampled_data = nullptr;

                // Allocate memory for the sampled data
                diskann::alloc_aligned((void**)&sampled_data, sampled_num_points * aligned_dim * sizeof(T), 8 * sizeof(T));

                // Open the binary file
                std::ifstream reader(data_path, std::ios::binary);
                if (!reader.is_open()) {
                    throw std::runtime_error("Failed to open file: " + data_path);
                }

                // Skip metadata (2 * sizeof(uint32_t))
                reader.seekg(2 * sizeof(uint32_t), std::ios::beg);

                // Randomly sample points
                std::default_random_engine generator(std::random_device{}());
                std::uniform_int_distribution<size_t> distribution(0, total_num_points - 1);

                std::unordered_set<size_t> sampled_indices;
                while (sampled_indices.size() < sampled_num_points) {
                    sampled_indices.insert(distribution(generator));
                }

                size_t current_index = 0;
                size_t sampled_index = 0;
                T* buffer = new T[aligned_dim];

                for (size_t i = 0; i < total_num_points; ++i) {
                    if (i % 100 == 0) {
                        std::cout << "Reading point " << i << "/" << total_num_points << "\r" << std::flush;
                    }
                    // Read the vector
                    reader.read(reinterpret_cast<char*>(buffer), dim * sizeof(T));
                    // Skip padding for aligned dimensions
                    reader.seekg((aligned_dim - dim) * sizeof(T), std::ios::cur);

                    // If the current index is in the sampled set, copy it to the sampled data
                    if (sampled_indices.count(i)) {
                        std::memcpy(sampled_data + sampled_index * aligned_dim, buffer, dim * sizeof(T));
                        std::memset(sampled_data + sampled_index * aligned_dim + dim, 0, (aligned_dim - dim) * sizeof(T));
                        ++sampled_index;
                    }

                    if (sampled_index >= sampled_num_points) {
                        break;
                    }
                }

                delete[] buffer;
                reader.close();
            }



        public:
            template <typename... Args>
            QVCache(const std::string& data_path,
                        const std::string& pca_prefix,
                        uint32_t R, uint32_t memory_L,
                        uint32_t B, uint32_t M,
                        float alpha,
                        uint32_t build_threads,
                        uint32_t search_threads,
                        bool use_reconstructed_vectors,
                        double p,
                        double deviation_factor, 
                        size_t memory_index_max_points,
                        uint32_t beamwidth_,
                        bool use_regional_theta = true,
                        size_t pca_dim = 16,
                        size_t buckets_per_dim = 4,
                        uint32_t n_async_insert_threads_ = 4,
                        bool lazy_theta_updates_ = true,
                        size_t number_of_mini_indexes_ = 2,
                        bool search_mini_indexes_in_parallel_ = false,
                        size_t max_search_threads_ = 32,
                        diskann::Metric metric_ = diskann::L2,
                        std::unique_ptr<BackendInterface<T, TagT>> disk_backend_ptr = nullptr)
                        : data_path(data_path),
                        pca_prefix(pca_prefix),
                        search_threads(search_threads),
                        use_reconstructed_vectors(use_reconstructed_vectors),
                        p(p),
                        deviation_factor(deviation_factor),
                        memory_L(memory_L),
                        beamwidth(beamwidth_),
                        use_regional_theta(use_regional_theta),
                        number_of_mini_indexes(number_of_mini_indexes_),
                        memory_index_max_points_per_index(memory_index_max_points / number_of_mini_indexes_), // Equal capacity per index
                        n_async_insert_threads(n_async_insert_threads_),
                        lazy_theta_updates(lazy_theta_updates_),
                        search_mini_indexes_in_parallel(search_mini_indexes_in_parallel_),
                        max_search_threads(max_search_threads_),
                        metric(metric_),
                        backend(std::move(disk_backend_ptr))
            {                
                // Read metadata
                diskann::get_bin_metadata(data_path, num_points, dim);
                aligned_dim = ROUND_UP(dim, 8);

                // Build LRU-managed memory indices
                memory_indices.reserve(number_of_mini_indexes);
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    memory_indices.push_back(create_memory_index(memory_index_max_points_per_index));
                }
                
                // Initialize LRU cache for managing mini-index access patterns
                lru_cache = std::make_unique<LRUCache<size_t>>(number_of_mini_indexes);
                
                // Set index 0 as active initially for insertions
                active_insert_index_id.store(0);
                
                // Initialize LRU cache with ALL indices (not just the active one)
                // This ensures we have a complete LRU order from the start
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    lru_cache->access(i);
                }
            
                std::cout << "QVCache LRU-managed memory indices built successfully!" << std::endl;
                std::cout << "Created " << number_of_mini_indexes << " indices, each can hold up to " << memory_index_max_points_per_index << " vectors" << std::endl;
                std::cout << "LRU eviction policy enabled" << std::endl;
                if (search_mini_indexes_in_parallel) {
                    std::cout << "Parallel search enabled with max " << max_search_threads << " threads" << std::endl;
                }

                std::cout << "QVCache disk index built successfully!" << std::endl;

                // Initialize insert thread pool for async insertions
                if (use_reconstructed_vectors) {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_reconstructed_sync(index, to_be_inserted);
                    };
                    if (lazy_theta_updates) {
                        auto theta_update_task = [this](T* query_ptr, uint32_t K, float query_distance) {
                            this->update_theta(query_ptr, K, query_distance);
                        };
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task, theta_update_task);
                    } else {
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                    }
                } else {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_sync(index, to_be_inserted);
                    };
                    if (lazy_theta_updates) {
                        auto theta_update_task = [this](T* query_ptr, uint32_t K, float query_distance) {
                            this->update_theta(query_ptr, K, query_distance);
                        };
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task, theta_update_task);
                    } else {
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                    }
                }

                std::cout << "QVCache built successfully with LRU eviction policy!" << std::endl;

                // PCA is constructed at construction time using Eigen. Eigen is required.
                if (use_regional_theta) {
                    pca_utils = std::make_unique<PCAUtils<T>>(dim, pca_dim, buckets_per_dim, pca_prefix, metric);
                    bool loaded = false;
                    if constexpr (std::is_floating_point<T>::value) {
                        loaded = pca_utils->load_pca_from_file(false);
                    } else {
                        loaded = pca_utils->load_pca_from_file(true);
                    }
                    if (loaded) {
                        std::cout << "[QVCache] Loaded PCA from file: " << pca_utils->get_pca_filename_for_logging() << std::endl;
                    } else {
                        std::cout << "[QVCache] No PCA file found or mismatch, running PCA..." << std::endl;
                        T* data = nullptr;
                        size_t sampled_num_points;
                        diskann::get_bin_metadata(data_path, num_points, dim);
                        aligned_dim = ROUND_UP(dim, 8);
                        load_sampled_data(data_path, data, sampled_num_points, aligned_dim, num_points);
                        std::cout << "[QVCache] Loaded " << sampled_num_points << " sampled points from " << data_path << std::endl;
                        pca_utils->construct_pca_from_data(data, sampled_num_points, aligned_dim, pca_prefix);
                        diskann::aligned_free(data);
                    }
                } else {
                    std::cout << "[QVCache] Skipping PCA construction (use_regional_theta is false)." << std::endl;
                }

                if (!use_regional_theta) {
                    // Initialize global theta_map for K=1,5,10,100
                    double init_value = (metric == diskann::COSINE) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::max();
                    theta_map[1] = init_value;
                    theta_map[5] = init_value;
                    theta_map[10] = init_value;
                    theta_map[100] = init_value;
                }
            }


            bool search(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                // Get current active insert index ID (atomic read)
                size_t current_active_id = active_insert_index_id.load();
                
                // Search all memory indices based on configured strategy
                bool is_hit = false;
                
                if (search_strategy == SearchStrategy::PARALLEL && search_mini_indexes_in_parallel && number_of_mini_indexes > 1) {
                    is_hit = parallel_search_memory_indices(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr);

                } else if (search_strategy == SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT) {
                    return search_sequential_lru_stop_first_hit(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                } else if (search_strategy == SearchStrategy::SEQUENTIAL_LRU_ADAPTIVE) {
                    return search_adaptive_hit_ratio(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                } else {
                    // Sequential search in LRU order (most recently used first)
                    std::vector<size_t> lru_order = lru_cache->get_all_tags();
                    
                    // Search indices in LRU order (most recently used first)
                    // But don't stop at first hit - search all indices to find the best result
                    bool found_hit = false;
                    float best_distance = std::numeric_limits<float>::max();
                    size_t best_index_id = 0;
                    std::vector<uint32_t> best_tags;
                    std::vector<float> best_dists;
                    std::vector<T*> best_res;
                    
                    for (size_t index_id : lru_order) {
                        if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                            // Allocate temporary storage for this search
                            std::vector<uint32_t> temp_tags(K);
                            std::vector<float> temp_dists(K);
                            std::vector<T*> temp_res;
                            
                            memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, temp_tags.data(), temp_dists.data(), temp_res);
                            bool is_hit = this->isHit(query_ptr, K, temp_dists.data());
                            
                            if (is_hit) {
                                found_hit = true;
                                // Check if this result is better than the current best
                                // Use the K-th distance as a quality metric
                                if (temp_dists[K-1] < best_distance) {
                                    best_distance = temp_dists[K-1];
                                    best_index_id = index_id;
                                    best_tags = std::move(temp_tags);
                                    best_dists = std::move(temp_dists);
                                    best_res = std::move(temp_res);
                                }
                            }
                        }
                    }
                    
                    if (found_hit) {
                        // Use the best result found
                        std::copy(best_tags.begin(), best_tags.end(), query_result_tags_ptr);
                        std::copy(best_dists.begin(), best_dists.end(), query_result_dists_ptr);
                        res = std::move(best_res);
                        
                        // Update LRU cache for the index that provided the best result
                        lru_cache->access(best_index_id);
                        
                        is_hit = true;
                    }
                }
                
                if (is_hit) {
                    return true; // Return true if the query is hit in the memory index
                }
                else {
                    
                    // Search disk using the backend interface
                    this->backend->search(query_ptr, (uint64_t)K, query_result_tags_ptr, query_result_dists_ptr, nullptr, backend_stats);
                    std::vector<uint32_t> tags_to_insert(query_result_tags_ptr, query_result_tags_ptr + K);
                    
                    if (lazy_theta_updates) {
                        // Copy query pointer for async insertion and theta update
                        T* query_copy = nullptr;
                        diskann::alloc_aligned((void**)&query_copy, this->aligned_dim * sizeof(T), 8 * sizeof(T));
                        std::memcpy(query_copy, query_ptr, this->aligned_dim * sizeof(T));
                        
                        // Submit to insert pool with current active index
                        insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1], query_copy);
                    } else {
                        // Immediate theta update in main thread
                        update_theta(query_ptr, K, query_result_dists_ptr[K - 1]);
                        insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1]);
                    }
                    
                    for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                    return false; // Return false if the query is missed in the memory index
                }
            }

            size_t get_number_of_vectors_in_memory_index() const {
                size_t total = 0;
                for (const auto& index : memory_indices) {
                    total += index->get_number_of_active_vectors();
                }
                return total;
            }

            size_t get_number_of_max_points_in_memory_index() const {
                return memory_index_max_points_per_index * number_of_mini_indexes; // Total capacity across all indices
            }

            // New methods for LRU eviction status
            bool is_eviction_in_progress() const {
                return eviction_in_progress.load();
            }

            size_t get_index_vector_count(size_t index_id) const {
                if (index_id < number_of_mini_indexes) {
                    return memory_indices[index_id]->get_number_of_active_vectors();
                }
                return 0;
            }

            size_t get_active_index_id() const {
                return active_insert_index_id.load();
            }

            size_t get_number_of_mini_indexes() const {
                return number_of_mini_indexes;
            }

            bool is_parallel_search_enabled() const {
                return search_mini_indexes_in_parallel;
            }

            size_t get_max_search_threads() const {
                return max_search_threads;
            }

            size_t get_number_of_active_pca_regions() const {
                if (use_regional_theta && pca_utils) {
                    return pca_utils->get_number_of_active_regions();
                }
                return 0;
            }

            // LRU management methods
            std::vector<size_t> get_lru_order() const {
                return lru_cache->get_all_tags();
            }

            std::vector<size_t> get_most_recently_used_indices(size_t n) const {
                return lru_cache->get_mru_tags(n);
            }

            std::vector<size_t> get_least_recently_used_indices(size_t n) const {
                return lru_cache->get_lru_tags(n);
            }

            size_t get_lru_cache_size() const {
                return lru_cache->size();
            }



            void set_search_strategy(SearchStrategy strategy) {
                search_strategy = strategy;
            }

            SearchStrategy get_search_strategy() const {
                return search_strategy;
            }
            
            // Hit ratio monitoring methods
            void update_hit_history(bool was_hit) {
                std::lock_guard<std::mutex> lock(hit_history_mutex);
                hit_history.push_back(was_hit);
                if (hit_history.size() > hit_ratio_window_size) {
                    hit_history.pop_front();
                }
            }
            
            double get_current_hit_ratio() const {
                std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(hit_history_mutex));
                if (hit_history.empty()) {
                    return 1.0;  // Assume good hit ratio if no history
                }
                size_t hits = std::count(hit_history.begin(), hit_history.end(), true);
                return static_cast<double>(hits) / hit_history.size();
            }
            
            bool should_use_adaptive_strategy() const {
                return use_adaptive_strategy && get_current_hit_ratio() < hit_ratio_threshold;
            }
            
            // Configuration methods for adaptive strategy
            void set_hit_ratio_window_size(size_t window_size) {
                std::lock_guard<std::mutex> lock(hit_history_mutex);
                hit_ratio_window_size = window_size;
                // Trim history if new window is smaller
                while (hit_history.size() > hit_ratio_window_size) {
                    hit_history.pop_front();
                }
            }
            
            void set_hit_ratio_threshold(double threshold) {
                hit_ratio_threshold = threshold;
            }
            
            void enable_adaptive_strategy(bool enable) {
                use_adaptive_strategy = enable;
            }
            


    };
}