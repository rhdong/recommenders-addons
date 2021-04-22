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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace recommenders_addons {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------
REGISTER_OP("TFRAUnique")
    .Input("x: T")
    .Output("y: T")
    .Output("idx: out_idx")
    .Attr("T: type")
    .Attr("out_idx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->input(0));
      // Assert that the input rank is 1.
      ShapeHandle dummy;
      return c->WithRank(c->input(0), 1, &dummy);
    });

REGISTER_OP("TFRAUniqueV2")
    .Input("x: T")
    .Input("axis: Taxis")
    .Output("y: T")
    .Output("idx: out_idx")
    .Attr("T: type")
    .Attr("Taxis: {int32,int64} = DT_INT64")
    .Attr("out_idx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(0))));
      c->set_output(1, c->input(0));
      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("TFRAUniqueWithCounts")
    .Input("x: T")
    .Output("y: T")
    .Output("idx: out_idx")
    .Output("count: out_idx")
    .Attr("T: type")
    .Attr("out_idx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      auto uniq = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, uniq);
      c->set_output(1, c->input(0));
      c->set_output(2, uniq);
      return Status::OK();
    });

REGISTER_OP("TFRAUniqueWithCountsV2")
    .Input("x: T")
    .Input("axis: Taxis")
    .Output("y: T")
    .Output("idx: out_idx")
    .Output("count: out_idx")
    .Attr("T: type")
    .Attr("Taxis: {int32,int64} = DT_INT64")
    .Attr("out_idx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(0))));
      c->set_output(1, c->input(0));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

}  // namespace recommenders_addons
}  // namespace tensorflow
