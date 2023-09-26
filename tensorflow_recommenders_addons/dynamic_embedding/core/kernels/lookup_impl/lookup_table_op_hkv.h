/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_

#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include <limits>
#include <string>
#include <typeindex>
#include <vector>

#include "merlin/allocator.cuh"
#include "merlin/types.cuh"
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
namespace gpu {

template <typename K, typename V, typename M>
class KVOnlyFile : public nv::merlin::BaseKVFile<K, V, M> {
 public:
  KVOnlyFile() : keys_fp_(nullptr), values_fp_(nullptr) {}

  ~KVOnlyFile() { close(); }

  bool open(const std::string& keys_path, const std::string& values_path,
            const char* mode) {
    close();
    keys_fp_ = fopen(keys_path.c_str(), mode);
    if (!keys_fp_) {
      return false;
    }
    values_fp_ = fopen(values_path.c_str(), mode);
    if (!values_fp_) {
      close();
      return false;
    }
    return true;
  }

  void close() noexcept {
    if (keys_fp_) {
      fclose(keys_fp_);
      keys_fp_ = nullptr;
    }
    if (values_fp_) {
      fclose(values_fp_);
      values_fp_ = nullptr;
    }
  }

  size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
              M* metas) override {
    size_t nread_keys =
        fread(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nread_vecs =
        fread(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    if (nread_keys != nread_vecs) {
      LOG(INFO) << "Partially read failed. " << nread_keys
                << " kv pairs by KVOnlyFile.";
      return 0;
    }
    LOG(INFO) << "Partially read " << nread_keys << " kv pairs by KVOnlyFile.";
    return nread_keys;
  }

  size_t write(const size_t n, const size_t dim, const K* keys,
               const V* vectors, const M* metas) override {
    size_t nwritten_keys =
        fwrite(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nwritten_vecs =
        fwrite(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    if (nwritten_keys != nwritten_vecs) {
      return 0;
    }
    LOG(INFO) << "Partially write " << nwritten_keys
              << " kv pairs by KVOnlyFile.";
    return nwritten_keys;
  }

 private:
  FILE* keys_fp_;
  FILE* values_fp_;
};

// template to avoid multidef in compile time only.
template <typename K, typename V>
__global__ void gpu_u64_to_i64_kernel(const uint64_t* u64, int64* i64,
                                      size_t len) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < len) {
    i64[tid] = static_cast<int64>(u64[tid]);
  }
}

template <typename T>
__global__ void broadcast_kernel(T* data, T val, size_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    data[tid] = val;
  }
}

template <typename K, typename V>
void gpu_cast_u64_to_i64(const uint64_t* u64, int64* i64, size_t len,
                         cudaStream_t stream) {
  size_t block_size = nv::merlin::SAFE_GET_BLOCK_SIZE(1024);
  size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size);
  gpu_u64_to_i64_kernel<K, V>
      <<<grid_size, block_size, 0, stream>>>(u64, i64, len);
}

using GPUDevice = Eigen::ThreadPoolDevice;

struct TableWrapperInitOptions {
  size_t max_capacity;
  size_t init_capacity;
  size_t max_hbm_for_vectors;
  size_t max_bucket_size;
  float max_load_factor;
  int block_size;
  int io_block_size;
};

template <typename V>
__global__ void gpu_fill_default_values(V* d_vals, V* d_def_val, size_t len,
                                        size_t dim) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < len * dim; i += stride) {
    int row = i / dim;
    int col = i % dim;
    d_vals[row * dim + col] = *d_def_val;
  }
}

class TFOrDefaultAllocator : public nv::merlin::BaseAllocator {
 private:
  using NMMemType = nv::merlin::MemoryType;
  // tensorflow::Allocator* tf_host_allocator_ = nullptr;
  tensorflow::Allocator* tf_device_allocator_ = nullptr;
  std::unique_ptr<nv::merlin::DefaultAllocator> default_allocator_ = nullptr;
  bool use_default_allocator_ = false;
  // bool tf_async_allocator_stream_set_ = false;
  static constexpr size_t kAllocatorAlignment = 8;

 public:
  TFOrDefaultAllocator() : use_default_allocator_(true) {
    default_allocator_ = std::make_unique<nv::merlin::DefaultAllocator>();
  }

  TFOrDefaultAllocator(OpKernelContext* ctx) {
    if (ctx) {
      tensorflow::AllocatorAttributes tf_alloc_attrs;
      tf_alloc_attrs.set_on_host(false);
      tf_device_allocator_ = ctx->get_allocator(tf_alloc_attrs);
    } else {
      use_default_allocator_ = true;
      default_allocator_ = std::make_unique<nv::merlin::DefaultAllocator>();
    }
  }

  ~TFOrDefaultAllocator() override {}

  void alloc(const NMMemType type, void** ptr, size_t size,
             unsigned int pinned_flags = cudaHostAllocDefault) override {
    if (!use_default_allocator_) {
      printf("!!!!not use_default_allocator_\n");
      switch (type) {
        case NMMemType::Device:
          printf("!!!!!!!!!!!!!!!!!!!!!!!!---------Device size: %zu\n", size);
          *ptr = tf_device_allocator_->AllocateRaw(kAllocatorAlignment, size);
          break;
        case NMMemType::Pinned:
          printf("!!!!!!!!!!!!!!!!!!!!!!!!---------Pinned size: %zu\n", size);
          CUDA_CHECK(cudaMallocHost(ptr, size, pinned_flags));
          break;
        case NMMemType::Host:
          printf("!!!!!!!!!!!!!!!!!!!!!!!!---------Host size: %zu\n", size);
          *ptr = std::malloc(size);
          break;
      }
    } else {
      printf("--->>>use_default_allocator_\n");
      default_allocator_->alloc(type, ptr, size, pinned_flags);
    }
  }

  void alloc_async(const NMMemType type, void** ptr, size_t size,
                   cudaStream_t stream) override {
    if (!use_default_allocator_) {
      if (NMMemType::Device == type) {
        *ptr = tf_device_allocator_->AllocateRaw(kAllocatorAlignment, size);
      }
    } else {
      default_allocator_->alloc_async(type, ptr, size, stream);
    }
  }

  void free(const NMMemType type, void* ptr) override {
    if (!use_default_allocator_) {
      switch (type) {
        case NMMemType::Device:
          tf_device_allocator_->DeallocateRaw(ptr);
          break;
        case NMMemType::Pinned:
          CUDA_CHECK(cudaFreeHost(ptr));
          break;
        case NMMemType::Host:
          std::free(ptr);
          break;
      }
    } else {
      default_allocator_->free(type, ptr);
    }
  }

  void free_async(const NMMemType type, void* ptr,
                  cudaStream_t stream) override {
    if (!use_default_allocator_) {
      if (NMMemType::Device == type) {
        tf_device_allocator_->DeallocateRaw(ptr);
      }
    } else {
      default_allocator_->free_async(type, ptr, stream);
    }
  }
};

template <class K, class V>
class TableWrapper {
 private:
  // using M = uint64_t;
  using Table = nv::merlin::HashTable<K, V, uint64_t>;
  nv::merlin::HashTableOptions mkv_options_;

 public:
  TableWrapper(TableWrapperInitOptions& init_options, size_t dim) {
    max_capacity_ = init_options.max_capacity;
    dim_ = dim;
    // nv::merlin::HashTableOptions mkv_options_;
    mkv_options_.init_capacity =
        std::min(init_options.init_capacity, max_capacity_);
    mkv_options_.max_capacity = max_capacity_;
    // Since currently GPU nodes are not compatible to fast
    // pcie connections for D2H non-continous wirte, so just
    // use pure hbm mode now.
    // mkv_options_.max_hbm_for_vectors = std::numeric_limits<size_t>::max();
    mkv_options_.max_hbm_for_vectors = init_options.max_hbm_for_vectors;
    mkv_options_.max_load_factor = 0.5;
    mkv_options_.block_size = nv::merlin::SAFE_GET_BLOCK_SIZE(1024);
    mkv_options_.dim = dim;
    mkv_options_.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
    block_size_ = mkv_options_.block_size;
    table_ = new Table();
  }

  Status init(nv::merlin::BaseAllocator* allocator) {
    try {
      table_->init(mkv_options_, allocator);
    } catch (std::runtime_error& e) {
      return Status(tensorflow::error::INTERNAL, e.what());
    }
    return Status::OK();
  }

  ~TableWrapper() { delete table_; }

  void upsert(const K* d_keys, const V* d_vals, size_t len,
              cudaStream_t stream) {
    uint64_t t0 = (uint64_t)time(NULL);
    uint64_t* timestamp_metas = nullptr;
    CUDA_CHECK(
        cudaMallocAsync(&timestamp_metas, len * sizeof(uint64_t), stream));
    CUDA_CHECK(
        cudaMemsetAsync(timestamp_metas, 0, len * sizeof(uint64_t), stream));
    size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size_);
    broadcast_kernel<uint64_t>
        <<<grid_size, block_size_, 0, stream>>>(timestamp_metas, t0, len);

    table_->insert_or_assign(len, d_keys, d_vals, /*d_metas=*/timestamp_metas,
                             stream);
    CUDA_CHECK(cudaFreeAsync(timestamp_metas, stream));
  }

  void accum(const K* d_keys, const V* d_vals_or_deltas, const bool* d_exists,
             size_t len, cudaStream_t stream) {
    uint64_t t0 = (uint64_t)time(NULL);
    uint64_t* timestamp_metas = nullptr;
    CUDA_CHECK(
        cudaMallocAsync(&timestamp_metas, len * sizeof(uint64_t), stream));
    CUDA_CHECK(
        cudaMemsetAsync(timestamp_metas, 0, len * sizeof(uint64_t), stream));
    size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size_);
    broadcast_kernel<uint64_t>
        <<<grid_size, block_size_, 0, stream>>>(timestamp_metas, t0, len);
    table_->accum_or_assign(len, d_keys, d_vals_or_deltas, d_exists,
                            /*d_metas=*/timestamp_metas, stream);
    CUDA_CHECK(cudaFreeAsync(timestamp_metas, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void dump(K* d_key, V* d_val, const size_t offset, const size_t search_length,
            size_t* d_dump_counter, cudaStream_t stream) const {
    table_->export_batch(search_length, offset, d_dump_counter, d_key, d_val,
                         /*d_metas=*/nullptr, stream);
  }

  void dump_with_metas(K* d_key, V* d_val, uint64_t* d_metas,
                       const size_t offset, const size_t search_length,
                       size_t* d_dump_counter, cudaStream_t stream) const {
    table_->export_batch(search_length, offset, d_dump_counter, d_key, d_val,
                         d_metas, stream);
  }

  void dump_keys_and_metas(K* keys, int64* metas, size_t len, size_t split_len,
                           cudaStream_t stream) const {
    V* values_buf = nullptr;
    size_t offset = 0;
    size_t real_offset = 0;
    size_t skip = split_len;
    uint64_t* metas_u64 = reinterpret_cast<uint64_t*>(metas);
    size_t span_len = table_->capacity();
    CUDA_CHECK(
        cudaMallocAsync(&values_buf, sizeof(V) * dim_ * split_len, stream));
    CUDA_CHECK(
        cudaMemsetAsync(values_buf, 0, sizeof(V) * dim_ * split_len, stream));
    for (; offset < span_len; offset += split_len) {
      if (offset + skip > span_len) {
        skip = span_len - offset;
      }
      // TODO: overlap the loop
      size_t h_dump_counter =
          table_->export_batch(skip, offset, keys + real_offset, values_buf,
                               metas_u64 + real_offset, stream);
      CudaCheckError();

      if (h_dump_counter > 0) {
        gpu_cast_u64_to_i64<K, V>(metas_u64 + real_offset, metas + real_offset,
                                  h_dump_counter, stream);
        real_offset += h_dump_counter;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(values_buf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // TODO (LinGeLin) support metas
  bool is_valid_metas(const std::string& keyfile,
                      const std::string& metafile) const {
    return false;
  }

  void dump_to_file(const string filepath, size_t dim, cudaStream_t stream,
                    const size_t buffer_size) const {
    LOG(INFO) << "dump_to_file, filepath: " << filepath << ", dim: " << dim
              << ", stream: " << stream << ", buffer_size: " << buffer_size;
    std::unique_ptr<nv::merlin::BaseKVFile<K, V, uint64_t>> wfile;
    string keyfile = filepath + "-keys";
    string valuefile = filepath + "-values";
    string metafile = filepath + "-metas";
    bool has_metas = false;
    bool open_ok = false;

    if (is_valid_metas(keyfile, metafile)) {
      wfile.reset(new nv::merlin::LocalKVFile<K, V, uint64_t>);
      open_ok = reinterpret_cast<nv::merlin::LocalKVFile<K, V, uint64_t>*>(
                    wfile.get())
                    ->open(keyfile, valuefile, metafile, "wb");
      has_metas = true;
    } else {
      wfile.reset(new KVOnlyFile<K, V, uint64_t>);
      open_ok = reinterpret_cast<KVOnlyFile<K, V, uint64_t>*>(wfile.get())
                    ->open(keyfile, valuefile, "wb");
    }
    if (!open_ok) {
      std::string error_msg = "Failed to dump to file to " + keyfile + ", " +
                              valuefile + ", " + metafile;
      throw std::runtime_error(error_msg);
    }

    size_t n_saved = table_->save(wfile.get(), buffer_size, stream);
    if (has_metas) {
      LOG(INFO) << "[op] Load " << n_saved << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile << ", and metafile"
                << metafile;
    } else {
      LOG(INFO) << "[op] Load " << n_saved << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // wfile->close();
  }

  void load_from_file(const string filepath, size_t key_num, size_t dim,
                      cudaStream_t stream, const size_t buffer_size) {
    std::unique_ptr<nv::merlin::BaseKVFile<K, V, uint64_t>> rfile;
    string keyfile = filepath + "-keys";
    string valuefile = filepath + "-values";
    string metafile = filepath + "-metas";
    // rfile.reset(new TimestampV1CompatFile<K, V, uint64_t>);
    bool has_metas = false;
    bool open_ok = false;

    if (is_valid_metas(keyfile, metafile)) {
      rfile.reset(new nv::merlin::LocalKVFile<K, V, uint64_t>);
      open_ok = reinterpret_cast<nv::merlin::LocalKVFile<K, V, uint64_t>*>(
                    rfile.get())
                    ->open(keyfile, valuefile, metafile, "rb");
      has_metas = true;
    } else {
      rfile.reset(new KVOnlyFile<K, V, uint64_t>);
      open_ok = reinterpret_cast<KVOnlyFile<K, V, uint64_t>*>(rfile.get())
                    ->open(keyfile, valuefile, "rb");
    }
    if (!open_ok) {
      std::string error_msg = "Failed to load from file " + keyfile + ", " +
                              valuefile + ", " + metafile;
      throw std::runtime_error(error_msg);
    }

    size_t n_loaded = table_->load(rfile.get(), buffer_size, stream);
    if (has_metas) {
      LOG(INFO) << "[op] Load " << n_loaded
                << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile << ", and metafile"
                << metafile;
    } else {
      LOG(INFO) << "[op] Load " << n_loaded
                << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (has_metas) {
      reinterpret_cast<nv::merlin::LocalKVFile<K, V, uint64_t>*>(rfile.get())
          ->close();
    } else {
      reinterpret_cast<KVOnlyFile<K, V, uint64_t>*>(rfile.get())->close();
    }
  }

  void get(const K* d_keys, V* d_vals, bool* d_status, size_t len, V* d_def_val,
           cudaStream_t stream, bool is_full_size_default) const {
    if (is_full_size_default) {
      CUDA_CHECK(cudaMemcpyAsync(d_vals, d_def_val, sizeof(V) * dim_ * len,
                                 cudaMemcpyDeviceToDevice, stream));
    } else {
      size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size_);
      gpu_fill_default_values<V>
          <<<grid_size, block_size_, dim_ * sizeof(V), stream>>>(
              d_vals, d_def_val, len, dim_);
    }
    table_->find(len, d_keys, d_vals, d_status, /*d_metas=*/nullptr, stream);
  }

  // TODO: Implement a contain kernel instead of find.
  void contains(const K* d_keys, V* d_status, size_t len, cudaStream_t stream) {
    // pass
    V* tmp_vals = nullptr;
    CUDA_CHECK(cudaMallocAsync(&tmp_vals, sizeof(V) * len * dim_, stream));
    CUDA_CHECK(cudaMemsetAsync(&tmp_vals, 0, sizeof(V) * len * dim_, stream));
    table_->find(len, d_keys, tmp_vals, d_status, /*d_metas=*/nullptr, stream);
    CUDA_CHECK(cudaFreeAsync(tmp_vals, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  size_t get_size(cudaStream_t stream) const { return table_->size(stream); }

  size_t get_capacity() const { return table_->capacity(); }

  void remove(const K* d_keys, size_t len, cudaStream_t stream) {
    table_->erase(len, d_keys, stream);
  }

  void clear(cudaStream_t stream) { table_->clear(stream); }

 private:
  Table* table_;
  size_t max_capacity_;
  size_t dim_;
  int block_size_;
  bool dynamic_mode_;
};

template <class K, class V>
Status CreateTableImpl(TableWrapper<K, V>** pptable,
                       TableWrapperInitOptions& options,
                       nv::merlin::BaseAllocator* allocator,
                       size_t runtime_dim) {
  *pptable = new TableWrapper<K, V>(options, runtime_dim);
  return (*pptable)->init(allocator);
}

}  // namespace gpu
}  // namespace lookup
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_
