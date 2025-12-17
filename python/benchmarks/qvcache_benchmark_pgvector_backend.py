#!/usr/bin/env python3
"""
Python version of qvcache_benchmark.cpp using pgvector backend

This script implements the window-based benchmark logic from the C++ version
but uses a Python-implemented pgvector backend.
"""

import argparse
import numpy as np
import json
import sys

# Import the compiled qvcache module
try:
    import qvcache as qvc
except ImportError:
    print("Error: qvcache module not found. Please build the Python bindings first.")
    print("Run: cd build && cmake .. && make")
    sys.exit(1)

from backends.pgvector_backend import PgVectorBackend
from benchmarks.utils import hybrid_search, calculate_recall, calculate_hit_recall, log_split_metrics


def experiment_benchmark(
    data_path: str,
    query_path: str,
    groundtruth_path: str,
    pca_prefix: str,
    R: int,
    memory_L: int,
    K: int,
    B: int,
    M: int,
    alpha: float,
    build_threads: int,
    search_threads: int,
    beamwidth: int,
    use_reconstructed_vectors: int,
    p: float,
    deviation_factor: float,
    memory_index_max_points: int,
    use_regional_theta: bool,
    pca_dim: int,
    buckets_per_dim: int,
    n_splits: int,
    n_split_repeat: int,
    n_async_insert_threads: int,
    lazy_theta_updates: bool,
    number_of_mini_indexes: int,
    search_mini_indexes_in_parallel: bool,
    max_search_threads: int,
    search_strategy: str,
    backend: PgVectorBackend,
    table_name: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "postgres",
    db_user: str = "postgres",
    db_password: str = "postgres",
    metric: str = "l2"
):
    """Run the window-based benchmark experiment with pgvector backend."""
    # Convert metric string to enum
    if metric.lower() == "cosine":
        metric_enum = qvc.Metric.COSINE
    elif metric.lower() == "l2":
        metric_enum = qvc.Metric.L2
    else:
        raise ValueError(f"Unsupported metric: {metric}. Supported: l2, cosine")
    
    # Create QVCache with pgvector backend
    qvcache = qvc.QVCache(
        data_path=data_path,
        pca_prefix=pca_prefix,
        R=R,
        memory_L=memory_L,
        B=B,
        M=M,
        alpha=alpha,
        build_threads=build_threads,
        search_threads=search_threads,
        use_reconstructed_vectors=bool(use_reconstructed_vectors),
        p=p,
        deviation_factor=deviation_factor,
        memory_index_max_points=memory_index_max_points,
        beamwidth=beamwidth,
        use_regional_theta=use_regional_theta,
        pca_dim=pca_dim,
        buckets_per_dim=buckets_per_dim,
        n_async_insert_threads=n_async_insert_threads,
        lazy_theta_updates=lazy_theta_updates,
        number_of_mini_indexes=number_of_mini_indexes,
        search_mini_indexes_in_parallel=search_mini_indexes_in_parallel,
        max_search_threads=max_search_threads,
        metric=metric_enum,
        backend=backend
    )
    
    # Set search strategy
    if search_strategy == "SEQUENTIAL_LRU_STOP_FIRST_HIT":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_STOP_FIRST_HIT)
    elif search_strategy == "SEQUENTIAL_LRU_ADAPTIVE":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_ADAPTIVE)
        qvcache.enable_adaptive_strategy(True)
        qvcache.set_hit_ratio_window_size(100)
        qvcache.set_hit_ratio_threshold(0.90)
    elif search_strategy == "SEQUENTIAL_ALL":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_ALL)
    elif search_strategy == "PARALLEL":
        qvcache.set_search_strategy(qvc.SearchStrategy.PARALLEL)
    else:
        print(f"Warning: Unknown search strategy '{search_strategy}', using default", file=sys.stderr)
    
    # Load ground truth
    groundtruth_ids, groundtruth_dists = qvc.load_ground_truth_data(groundtruth_path)
    n_groundtruth, groundtruth_dim = groundtruth_ids.shape

    # Load queries
    queries, query_dim, query_aligned_dim = qvc.load_aligned_binary_data(query_path)
    query_num = queries.shape[0]
    
    # Ensure queries are float32
    queries = queries.astype(np.float32)
    
    # Query file structure: all copies of split 0, then all copies of split 1, etc.
    # Each split has n_split_repeat copies, and each copy has queries_per_original_split queries
    queries_per_original_split = query_num // (n_splits * n_split_repeat)
    
    # Process splits one by one sequentially
    for split_idx in range(n_splits):
        print(json.dumps({
            "event": "split_start",
            "split_idx": split_idx
        }))
        
        # Process all copies for this split
        for copy_idx in range(n_split_repeat):
            # Calculate query range for this specific copy of this split
            # Structure: split 0 (all copies), split 1 (all copies), ...
            # For split i, copy j: offset = i * (n_split_repeat * queries_per_original_split) + j * queries_per_original_split
            split_offset = split_idx * n_split_repeat * queries_per_original_split
            copy_offset = copy_idx * queries_per_original_split
            query_start = split_offset + copy_offset
            query_end = min(query_start + queries_per_original_split, query_num)
            
            if query_start >= query_end:
                continue
            
            this_split_size = query_end - query_start
            split_queries = queries[query_start:query_end]
            
            hit_results, _, query_result_tags, metrics = hybrid_search(
                qvcache,
                split_queries,
                K,
                search_threads,
                data_path
            )
            
            # Calculate groundtruth offset (same structure as queries)
            gt_start = split_offset + copy_offset
            recall_all = calculate_recall(
                K, groundtruth_ids[gt_start:gt_start + this_split_size], 
                query_result_tags, this_split_size, groundtruth_dim
            )
            recall_hits = calculate_hit_recall(
                K, groundtruth_ids[gt_start:gt_start + this_split_size], 
                query_result_tags, hit_results, this_split_size, groundtruth_dim
            )
            
            log_split_metrics(metrics, recall_all, recall_hits, split_idx=split_idx)
        
        print(json.dumps({
            "event": "split_end",
            "split_idx": split_idx
        }))
    
    # Give async insert threads time to complete before cleanup
    # QVCache uses async insert threads that might still be processing
    import time
    time.sleep(2)  # Wait 2 seconds for async operations to complete
    
    # Explicitly delete qvcache to trigger cleanup
    del qvcache


def main():
    parser = argparse.ArgumentParser(description="QVCache benchmark experiment with pgvector backend")
    parser.add_argument("--data_path", type=str, required=True, help="Path to base data file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query data file")
    parser.add_argument("--groundtruth_path", type=str, required=True, help="Path to groundtruth file")
    parser.add_argument("--pca_prefix", type=str, required=True, help="PCA index prefix")
    parser.add_argument("--table_name", type=str, default="vectors",
                       help="PostgreSQL table name (should match dataset name, default: vectors)")
    parser.add_argument("--db_host", type=str, default="localhost",
                       help="PostgreSQL service host (default: localhost)")
    parser.add_argument("--db_port", type=int, default=5432,
                       help="PostgreSQL service port (default: 5432)")
    parser.add_argument("--db_name", type=str, default="postgres",
                       help="Database name (default: postgres)")
    parser.add_argument("--db_user", type=str, default="postgres",
                       help="Database user (default: postgres)")
    parser.add_argument("--db_password", type=str, default="postgres",
                       help="Database password (default: postgres)")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine"],
                       help="Distance metric (default: l2)")
    
    # QVCache parameters
    parser.add_argument("--R", type=int, default=64, help="R parameter")
    parser.add_argument("--memory_L", type=int, default=128, help="Memory L parameter")
    parser.add_argument("--K", type=int, default=100, help="Number of nearest neighbors")
    parser.add_argument("--B", type=int, default=8, help="B parameter")
    parser.add_argument("--M", type=int, default=8, help="M parameter")
    parser.add_argument("--alpha", type=float, default=1.2, help="Alpha parameter")
    parser.add_argument("--build_threads", type=int, default=8, help="Build threads")
    parser.add_argument("--search_threads", type=int, default=24, help="Search threads")
    parser.add_argument("--beamwidth", type=int, default=2, help="Beamwidth")
    parser.add_argument("--use_reconstructed_vectors", type=int, default=0, help="Use reconstructed vectors")
    parser.add_argument("--p", type=float, default=0.9, help="P parameter")
    parser.add_argument("--deviation_factor", type=float, default=0.025, help="Deviation factor")
    parser.add_argument("--memory_index_max_points", type=int, default=200000, help="Max points in memory index")
    parser.add_argument("--use_regional_theta", type=bool, default=True, help="Use regional theta")
    parser.add_argument("--pca_dim", type=int, default=16, help="PCA dimension")
    parser.add_argument("--buckets_per_dim", type=int, default=8, help="Buckets per dimension")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits for queries")
    parser.add_argument("--n_split_repeat", type=int, required=True, help="Number of repeats per split pattern")
    parser.add_argument("--n_async_insert_threads", type=int, default=16, help="Async insert threads")
    parser.add_argument("--lazy_theta_updates", type=bool, default=True, help="Lazy theta updates")
    parser.add_argument("--number_of_mini_indexes", type=int, default=4, help="Number of mini indexes")
    parser.add_argument("--search_mini_indexes_in_parallel", type=bool, default=False, help="Search mini indexes in parallel")
    parser.add_argument("--max_search_threads", type=int, default=32, help="Max search threads")
    parser.add_argument("--search_strategy", type=str, default="SEQUENTIAL_LRU_ADAPTIVE",
                       choices=["SEQUENTIAL_LRU_STOP_FIRST_HIT", "SEQUENTIAL_LRU_ADAPTIVE",
                               "SEQUENTIAL_ALL", "PARALLEL"],
                       help="Search strategy")
    parser.add_argument("--data_type", type=str, default="float", help="Data type")
    
    args = parser.parse_args()
    
    # Print parameters
    params = {
        "event": "params",
        "backend": "pgvector",
        "table_name": args.table_name,
        "db_host": args.db_host,
        "db_port": args.db_port,
        "db_name": args.db_name,
        "db_user": args.db_user,
        "data_type": args.data_type,
        "data_path": args.data_path,
        "query_path": args.query_path,
        "groundtruth_path": args.groundtruth_path,
        "pca_prefix": args.pca_prefix,
        "R": args.R,
        "memory_L": args.memory_L,
        "K": args.K,
        "B": args.B,
        "M": args.M,
        "build_threads": args.build_threads,
        "search_threads": args.search_threads,
        "alpha": args.alpha,
        "use_reconstructed_vectors": args.use_reconstructed_vectors,
        "beamwidth": args.beamwidth,
        "p": args.p,
        "deviation_factor": args.deviation_factor,
        "use_regional_theta": args.use_regional_theta,
        "pca_dim": args.pca_dim,
        "buckets_per_dim": args.buckets_per_dim,
        "memory_index_max_points": args.memory_index_max_points,
        "n_splits": args.n_splits,
        "n_split_repeat": args.n_split_repeat,
        "n_async_insert_threads": args.n_async_insert_threads,
        "lazy_theta_updates": args.lazy_theta_updates,
        "number_of_mini_indexes": args.number_of_mini_indexes,
        "search_mini_indexes_in_parallel": args.search_mini_indexes_in_parallel,
        "max_search_threads": args.max_search_threads,
        "search_strategy": args.search_strategy,
        "metric": args.metric
    }
    print(json.dumps(params))
    
    # Get vector dimension from data file
    with open(args.data_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        dimension = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])  # Convert to Python int
    
    # Create pgvector backend (assumes table already exists)
    print(f"\nConnecting to PostgreSQL at {args.db_host}:{args.db_port}...", file=sys.stderr)
    backend = PgVectorBackend(
        table_name=args.table_name,
        dimension=dimension,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        data_path=None,  # Don't load data here, should already be indexed
        recreate_table=False,
        metric=args.metric
    )
    
    # Run experiment
    experiment_benchmark(
        args.data_path, args.query_path, args.groundtruth_path, args.pca_prefix,
        args.R, args.memory_L, args.K, args.B, args.M, args.alpha,
        args.build_threads, args.search_threads, args.beamwidth,
        args.use_reconstructed_vectors, args.p, args.deviation_factor,
        args.memory_index_max_points,
        args.use_regional_theta, args.pca_dim, args.buckets_per_dim,
        args.n_splits, args.n_split_repeat,
        args.n_async_insert_threads,
        args.lazy_theta_updates, args.number_of_mini_indexes,
        args.search_mini_indexes_in_parallel, args.max_search_threads,
        args.search_strategy, backend, args.table_name, args.db_host, args.db_port,
        args.db_name, args.db_user, args.db_password, args.metric
    )


if __name__ == "__main__":
    main()

