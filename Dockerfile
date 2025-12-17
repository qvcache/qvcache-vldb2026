FROM ubuntu:jammy

# Install system dependencies
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
        git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev \
        clang-format libboost-dev libboost-program-options-dev \
        libmkl-full-dev libcpprest-dev python3.10 python3.10-dev python3-pip \
        libeigen3-dev \
        libspdlog-dev && \
    # Install Python dependencies for bindings
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "protobuf<5.0.0" && \
    python3 -m pip install pybind11 numpy matplotlib qdrant-client pinecone psycopg2-binary faiss-cpu

WORKDIR /app
