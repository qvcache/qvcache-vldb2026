# QVCache: A Query-Aware Vector Cache

## Table of Contents

- [Installation](#installation)
- [Preparing Data](#preparing-data)
- [Reproducing Results](#reproducing-results)

## Installation

The easiest way to set up QVCache is using Docker containers, which handle all dependencies automatically.

1. **Start the containers**:
   ```bash
   docker-compose up -d
   ```
   **Note**: If you're using Docker Compose V2, you can also use `docker compose up -d` (without hyphen).
   
   This creates three containers:
   - `qvcache`: Main QVCache container
   - `qdrant`: Qdrant vector database 
   - `postgres`: PostgreSQL with pgvector extension 

2. **Connect to the QVCache container**:
   ```bash
   docker ps  # Find the qvcache container ID
   docker exec -it <qvcache-container-id> bash
   ```

3. **Build the project**:
   ```bash
   ./build.sh
   ```
   This builds both C++ binaries and Python bindings.

## Preparing Data

We provide the siftsmall dataset (http://corpus-texmex.irisa.fr/) under the `data/` folder. If you follow the same naming convention and folder structure for other datasets, you can simply repeat the experiments by changing the dataset name and parameters in the bash scripts.

### Converting fvecs/ivecs to .bin

Scripts accept datasets in `.bin` format. As an example, you can transform the related files from the siftsmall dataset as follows:

```bash
./build/utils/fvecs_to_bin float data/siftsmall/siftsmall_base.fvecs data/siftsmall/siftsmall_base.bin
./build/utils/fvecs_to_bin float data/siftsmall/siftsmall_query.fvecs data/siftsmall/siftsmall_query.bin
./build/utils/ivecs_to_bin data/siftsmall/siftsmall_groundtruth.ivecs data/siftsmall/siftsmall_groundtruth.bin
```

### Generating queries with temporal-semantic locality

You can run the following script to generate queries for the experiments:

```bash
./scripts/workload_generation/generate_noisy_queries.sh
```

## Reproducing Results

### Experiments in Figure 4, 6, 7 and 8

#### 1. Building a DiskANN index

You can run the following command to build a DiskANN index. It creates a folder with the same name as the dataset to store index files.

```bash
./scripts/diskann/build_disk_index.sh
```

#### 2. Running Experiments

##### 2.1. Backend-Only Setup

You can run the following script to get the results for the backend-only setting:

```bash
./scripts/qvcache/backend_benchmark.sh
```

##### 2.2. With QVCache Setup

You can run the following script to get the results for the setup in which DiskANN and QVCache work together:

```bash
./scripts/qvcache/qvcache_benchmark.sh
```

### Experiments in Figure 5

You should build the corresponding DiskANN index as described in section 1 above to run the following experiments.

#### 2. Running Experiments

##### 2.1. Backend-Only Setup

You can run the following script to get the results for the backend-only setting:

```bash
./scripts/qvcache/backend_cache_pressure.sh
```

##### 2.2. With QVCache Setup

You can run the following script to get the results for the setup in which DiskANN and QVCache work together:

```bash
./scripts/qvcache/cache_pressure.sh
```

---

### Experiments in Figure 8

#### 1. Building indexes for different backends

You can build indexes for different backends using the following scripts:

```bash
./python/scripts/build_index/build_faiss_index.sh
./python/scripts/build_index/build_pgvector_index.sh
./python/scripts/build_index/build_pinecone_index.sh
./python/scripts/build_index/build_qdrant_index.sh
```

#### 2. Running Experiments

##### 2.1. Backend-Only Setup

You can run the following scripts to get results for backend-only settings with different backends:

```bash
./python/scripts/benchmark/backend_only_benchmark_faiss_backend.sh
./python/scripts/benchmark/backend_only_benchmark_pgvector_backend.sh
./python/scripts/benchmark/backend_only_benchmark_pinecone_backend.sh
./python/scripts/benchmark/backend_only_benchmark_qdrant_backend.sh
```

##### 2.2. With QVCache Setup

You can run the following scripts to get results for setups in which QVCache works with different backends:

```bash
./python/scripts/benchmark/qvcache_benchmark_faiss_backend.sh
./python/scripts/benchmark/qvcache_benchmark_pgvector_backend.sh
./python/scripts/benchmark/qvcache_benchmark_pinecone_backend.sh
./python/scripts/benchmark/qvcache_benchmark_qdrant_backend.sh
``` 

**Note**: All bash scripts provided in this repository are self-explanatory and well-documented. The metrics are directed to standard output, and you can use the logs for visualization and analysis. You can edit the scripts to try different datasets and parameter values as needed. 
