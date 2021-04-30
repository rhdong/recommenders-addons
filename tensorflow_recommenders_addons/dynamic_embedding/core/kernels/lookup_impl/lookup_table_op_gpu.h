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

#include <typeindex>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/nvhash/nv_hashtable.cuh"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Lookup table op that supports different table implementations specified by
// the 'Container' template. Container must be derived from LookupInterface. The
// key and value are of the templated type "key_dtype" and "value_dtype"
// respectively.
template <class Container, class key_dtype, class value_dtype>
class LookupTableOp : public OpKernel {
 public:
  // ctx is not owned by this class.
  explicit LookupTableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  // ctx is not owned by this function.
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    if (!table_handle_set_) {
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_on_host(true);
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_persistent(tensorflow::DT_STRING,
                                              tensorflow::TensorShape({2}),
                                              &table_handle_, nullptr, attr));

      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator =
        [ctx, this](lookup::LookupInterface** ret)
            TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              lookup::LookupInterface* container = new Container(ctx, this);
              if (!ctx->status().ok()) {
                container->Unref();
                return ctx->status();
              }
              if (ctx->track_allocations()) {
                ctx->record_persistent_memory_allocation(
                    container->MemoryUsed() + table_handle_.AllocatedBytes());
              }
              *ret = container;
              return Status::OK();
            };

    lookup::LookupInterface* table = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()
                       ->template LookupOrCreate<lookup::LookupInterface>(
                           cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, lookup::CheckTableDataTypes(
                            *table, DataTypeToEnum<key_dtype>::v(),
                            DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      Tensor* handle;
#if GOOGLE_CUDA
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_on_host(true);
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, TensorShape({}), &handle, attr));
#else
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
#endif
      handle->scalar<ResourceHandle>()() =
          MakeResourceHandle<lookup::LookupInterface>(ctx, cinfo_.container(),
                                                      cinfo_.name());
    } else {
      if (!table_handle_set_) {
        auto h = table_handle_.AccessTensor(ctx)->template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      ctx->set_output_ref(0, &mu_, table_handle_.AccessTensor(ctx));
    }
    table_handle_set_ = true;
  }

  ~LookupTableOp() override {
    // If the table object was not shared, delete it.
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<lookup::LookupInterface>(cinfo_.container(),
                                                          cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

 private:
  mutex mu_;
  PersistentTensor table_handle_ TF_GUARDED_BY(mu_);
  bool table_handle_set_ TF_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(LookupTableOp);
};

namespace recommenders_addons {
namespace lookup {
namespace gpu {

using GPUDevice = Eigen::ThreadPoolDevice;

template <class V>
struct ValueArrayBase {};

template <class V, size_t DIM>
struct ValueArray : public ValueArrayBase<V> {
  V value[DIM];
};

template <class T>
using ValueType = ValueArrayBase<T>;

template <class K, class V>
class TableWrapperBase {
 public:
  virtual ~TableWrapperBase() {}
  virtual void upsert(const K* d_keys, const ValueType<V>* d_vals, size_t len,
                      cudaStream_t stream) {}
  virtual void dump(K* d_key, ValueType<V>* d_val, const size_t offset,
                    const size_t search_length, size_t* d_dump_counter,
                    cudaStream_t stream) const {}
  virtual void get(const K* d_keys, ValueType<V>* d_vals, bool* d_status,
                   size_t len, ValueType<V>* d_def_val, cudaStream_t stream,
                   bool is_full_size_default) const {}
  virtual size_t get_size(cudaStream_t stream) const {}
  virtual size_t get_capacity() const {}
  virtual void remove(const K* d_keys, size_t len, cudaStream_t stream) {}
};

template <class K, class V, size_t DIM>
class TableWrapper final : public TableWrapperBase<K, V> {
 private:
  using Table = nv::HashTable<K, ValueArray<V, DIM>, ValueType<V>,
                              std::numeric_limits<K>::max()>;

 public:
  TableWrapper(size_t max_size) : max_size_(max_size) {
    table_ = new Table(max_size);
  }

  ~TableWrapper() override { delete table_; }

  void upsert(const K* d_keys, const ValueType<V>* d_vals, size_t len,
              cudaStream_t stream) override {
    table_->upsert(d_keys, d_vals, len, stream);
  }

  void dump(K* d_key, ValueType<V>* d_val, const size_t offset,
            const size_t search_length, size_t* d_dump_counter,
            cudaStream_t stream) const override {
    table_->dump(d_key, d_val, offset, search_length, d_dump_counter, stream);
  }

  void get(const K* d_keys, ValueType<V>* d_vals, bool* d_status, size_t len,
           ValueType<V>* d_def_val, cudaStream_t stream,
           bool is_full_size_default) const override {
    table_->get(d_keys, d_vals, d_status, len, d_def_val, stream,
                is_full_size_default);
  }

  size_t get_size(cudaStream_t stream) const override {
    return table_->get_size(stream);
  }

  size_t get_capacity() const override { return table_->get_capacity(); }

  void remove(const K* d_keys, size_t len, cudaStream_t stream) override {
    table_->remove(d_keys, len, stream);
  }

 private:
  size_t max_size_;
  Table* table_;
};

#define CREATE_A_TABLE(DIM)                                   \
  do {                                                        \
    if (runtime_dim == (DIM + 1)) {                           \
      *pptable = new TableWrapper<K, V, (DIM + 1)>(max_size); \
    };                                                        \
  } while (0)

#define CREATE_TABLE_PARTIAL_BRANCHES(PERIFX) \
  do {                                        \
    CREATE_A_TABLE((PERIFX)*10 + 0);          \
    CREATE_A_TABLE((PERIFX)*10 + 1);          \
    CREATE_A_TABLE((PERIFX)*10 + 2);          \
    CREATE_A_TABLE((PERIFX)*10 + 3);          \
    CREATE_A_TABLE((PERIFX)*10 + 4);          \
    CREATE_A_TABLE((PERIFX)*10 + 5);          \
    CREATE_A_TABLE((PERIFX)*10 + 6);          \
    CREATE_A_TABLE((PERIFX)*10 + 7);          \
    CREATE_A_TABLE((PERIFX)*10 + 8);          \
    CREATE_A_TABLE((PERIFX)*10 + 9);          \
  } while (0)

// create branches with dim range:
// [CENTILE * 100 + (DECTILE) * 10, CENTILE * 100 + (DECTILE) * 10 + 50]
#define CREATE_TABLE_BRANCHES(CENTILE, DECTILE)              \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 0); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 1); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 2); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 3); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 4);

template <class K, class V, int centile, int dectile>
void CreateTableImpl(TableWrapperBase<K, V>** pptable, size_t max_size,
                     size_t runtime_dim) {
  CREATE_TABLE_BRANCHES(centile, dectile);
}

#define DEFINE_CREATE_TABLE(ID, K, V, CENTILE, DECTILE)                      \
  void CreateTable##ID(size_t max_size, size_t runtime_dim,                  \
                       TableWrapperBase<K, V>** pptable) {                   \
    CreateTableImpl<K, V, CENTILE, DECTILE>(pptable, max_size, runtime_dim); \
  }

#define DECLARE_CREATE_TABLE(K, V)                       \
  void CreateTable0(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable1(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable2(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable3(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable4(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable5(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable6(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable7(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable8(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable9(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);

DECLARE_CREATE_TABLE(int64, float);
DECLARE_CREATE_TABLE(int64, Eigen::half);
DECLARE_CREATE_TABLE(int64, int32);
DECLARE_CREATE_TABLE(int64, int8);

#undef CREATE_A_TABLE
#undef CREATE_DEFAULT_TABLE
#undef CREATE_TABLE_PARTIAL_BRANCHES
#undef CREATE_TABLE_ALL_BRANCHES
#undef DECLARE_CREATE_TABLE

}  // namespace gpu
}  // namespace lookup
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_
