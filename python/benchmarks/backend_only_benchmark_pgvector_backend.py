#!/usr/bin/env python3
"""
Backend-only benchmark using PgVector backend (without QVCache)

This script benchmarks the PgVector backend directly, processing splits sequentially.
"""

import argparse
import numpy as np
import json
import sys

try:
    import qvcache as qvc
except ImportError:
    print("Error: qvcache module not found. Please build the Python bindings first.")
    print("Run: cd build && cmake .. && make")
    sys.exit(1)

from backends.pgvector_backend import PgVectorBackend
from benchmarks.utils import backend_search, calculate_backend_recall, log_backend_split_metrics


def experiment_benchmark(
    data_path: str,
    query_path: str,
    groundtruth_path: str,
    K: int,
    search_threads: int,
    n_splits: int,
    n_split_repeat: int,
    backend: PgVectorBackend,
    table_name: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "postgres"
):
    """Run the backend-only benchmark experiment with PgVector backend."""
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
            
            query_result_tags, metrics = backend_search(
                backend,
                split_queries,
                K,
                search_threads
            )
            
            # Calculate groundtruth offset (same structure as queries)
            gt_start = split_offset + copy_offset
            recall_all = calculate_backend_recall(
                K, groundtruth_ids[gt_start:gt_start + this_split_size], 
                query_result_tags, this_split_size, groundtruth_dim
            )
            
            log_backend_split_metrics(metrics, recall_all, split_idx=split_idx)
        
        print(json.dumps({
            "event": "split_end",
            "split_idx": split_idx
        }))


def main():
    parser = argparse.ArgumentParser(description="Backend-only benchmark experiment with PgVector backend")
    parser.add_argument("--data_path", type=str, required=True, help="Path to base data file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query data file")
    parser.add_argument("--groundtruth_path", type=str, required=True, help="Path to groundtruth file")
    parser.add_argument("--table_name", type=str, default="vectors", help="PostgreSQL table name")
    parser.add_argument("--db_host", type=str, default="localhost", help="PostgreSQL host")
    parser.add_argument("--db_port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--db_name", type=str, default="postgres", help="PostgreSQL database name")
    parser.add_argument("--db_user", type=str, default="postgres", help="PostgreSQL user")
    parser.add_argument("--db_password", type=str, default="postgres", help="PostgreSQL password")
    
    # Search parameters
    parser.add_argument("--K", type=int, default=100, help="Number of nearest neighbors")
    parser.add_argument("--search_threads", type=int, default=24, help="Search threads")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits for queries")
    parser.add_argument("--n_split_repeat", type=int, required=True, help="Number of repeats per split pattern")
    parser.add_argument("--data_type", type=str, default="float", help="Data type")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine", "inner_product"],
                       help="Distance metric: l2, cosine, or inner_product")
    
    args = parser.parse_args()
    
    # Print parameters
    params = {
        "event": "params",
        "backend": "PgVector",
        "table_name": args.table_name,
        "db_host": args.db_host,
        "db_port": args.db_port,
        "db_name": args.db_name,
        "data_type": args.data_type,
        "data_path": args.data_path,
        "query_path": args.query_path,
        "groundtruth_path": args.groundtruth_path,
        "K": args.K,
        "search_threads": args.search_threads,
        "n_splits": args.n_splits,
        "n_split_repeat": args.n_split_repeat,
        "metric": args.metric
    }
    print(json.dumps(params))
    
    # Get vector dimension from data file
    with open(args.data_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        dimension = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    
    # Create PgVector backend (assumes table already exists)
    print(f"\nConnecting to PostgreSQL at {args.db_host}:{args.db_port}...", file=sys.stderr)
    backend = PgVectorBackend(
        table_name=args.table_name,
        dimension=dimension,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        data_path=args.data_path,
        recreate_table=False
    )
    
    # Run experiment
    experiment_benchmark(
        args.data_path, args.query_path, args.groundtruth_path,
        args.K, args.search_threads,
        args.n_splits, args.n_split_repeat,
        backend, args.table_name, args.db_host, args.db_port, args.db_name
    )


if __name__ == "__main__":
    main()

