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

#include "tensorflow/compiler/tf2tensorrt/convert/ops/slice_ops.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <bitset>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/strided_slice_op.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Adds a set of operations to the network which set the parameters for the
// given "slice_layer" in order to handle dynamic input shape.
Status HandleDynamicStridedSliceInput(
    TRTNetworkBuilder* builder, nvinfer1::ISliceLayer* slice_layer,
    const StridedSliceShapeSpec& strided_slice_spec,
    const absl::InlinedVector<int64, 4>& dynamic_input_size_indices,
    nvinfer1::Dims begin_dims, nvinfer1::Dims stride_dims,
    nvinfer1::Dims end_dims);

int ReturnValidBeginIndex(const OpConverterParams * params, const TRT_TernsorOrWeights& input,
                       const int index,
                       const vector<int> input_shape){
  //Check the index is in the valid input range using input shape
  
  // Get the shape during runtime.

  // Make a range of values.
  // begin is valid from -N to N-1 where N is the shape at a specific dimension.

  //check that the index is in the range and clamp if necessary.



}

int ReturnValidEndIndex(const OpConverterParams * params, const TRT_TernsorOrWeights& input,
                       const int index,
                       const vector<int> input_shape){
  // Check the index is in the valid input range using input shape.

  // Get the shape during runtime.

  // Make a range of values.
  // end can vary -(N-1) to (N) where N is the shape at a specific dimension

  // Check the index and clamp if necessary.
  // if end < -(N+1), then end = 0
  // if end > (N), then end = N


}
Status ConvertStridedSliceHelper(
    const OpConverterParams* params, const TRT_TensorOrWeights& input,
    const PartialTensorShape& input_dims, const SliceDims& begin,
    const SliceDims& stride, const SliceDims& end,
    std::optional<nvinfer1::Dims> final_shape, std::optional<int> op_instance,
    std::optional<StridedSliceShapeSpec> strided_slice_spec) {
  const auto& node_def = params->node_def;

  auto begin_dims = DimsAdapter::Create(begin, params->use_implicit_batch);
  auto stride_dims = DimsAdapter::Create(stride, params->use_implicit_batch);
  auto end_dims = DimsAdapter::Create(end, params->use_implicit_batch);
  TRT_ENSURE_OK(begin_dims);
  TRT_ENSURE_OK(stride_dims);
  TRT_ENSURE_OK(end_dims);

  // For each dimension, gather information about static vs dynamic dimension
  // and slice size.
  nvinfer1::Dims dims_before = input->trt_tensor()->getDimensions();
  nvinfer1::Dims size_dims = begin_dims->AsTrtDims();
  absl::InlinedVector<int64, 4> static_input_size_indices;
  absl::InlinedVector<int64, 4> dynamic_input_size_indices; 
  absl::InlinedVector<int64, 4> process_begin_indices; 
  absl::InlinedVector<int64, 4> process_end_indices; 

  for (int idx = 0; idx  < dims_before.nbDims; idx++) {
    size_dims.d[idx] = (std::abs(end_dims->dim(idx) - begin_dims->dim(idx)) +
                      std::abs(stride_dims->dim(idx)) - 1) /
                     std::abs(stride_dims->dim(idx));
    if (input_dims.dim_size(idx) == -1) {
      // end_dims and begin_dims do not have valid information yet.
      dynamic_input_size_indices.push_back(idx);
      // add end_dims for processing when input_dims are unknown 
      // incase the end_dims are out of range. 
      process_end_indices.push_back(idx);
    } else {
      static_input_size_indices.push_back(idx);
      if (end_dims->dim(idx) < begin_dims->dim(idx) && stride_dims->dim(idx) > 0) {
        return errors::InvalidArgument(
            "\"size\" cannot be negative for StridedSlice");
      }
    }
    //below standard range
    if(begin_dims->dim(idx) < 0) process_begin_indices.push_back(idx);
    if(end_dims->dim(idx) < 0) process_end_indices.push_back(idx);
     
  }

  if (!dynamic_input_size_indices.empty()) {
    if (strided_slice_spec == std::nullopt) {
      return errors::InvalidArgument(
          "The argument `strided_slice_spec` is "
          "`std::nullopt` with `dynamic_input_size_indices` non empty.");
    }
    if (params->use_implicit_batch) {
      return errors::InvalidArgument(
          "In implicit batch mode, dynamic input size is not supported.");
    }
  }

  if (params->validation_only) return OkStatus();

  StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
      params->converter->network(), params->weight_store);
  TRT_ENSURE_OK(builder);

  // Create the slice operation. For dynamic dims, the inputs of the operations
  // may be reassigned later.
  StatusOr<nvinfer1::ISliceLayer*> slice =
      builder->Slice(input.tensor()->trt_tensor(), begin_dims->AsTrtDims(),
                     size_dims, stride_dims->AsTrtDims());
  TRT_ENSURE_PTR_OK(slice);

  // Handle dynamic input shapes.
  if (!dynamic_input_size_indices.empty()) {
    TF_RETURN_IF_ERROR(HandleDynamicStridedSliceInput(
        &*builder, *slice, *strided_slice_spec, dynamic_input_size_indices,
        begin_dims->AsTrtDims(), stride_dims->AsTrtDims(),
        end_dims->AsTrtDims()));
  }

  params->converter->SetLayerName(*slice, params->node_def, "slice",
                                  op_instance);
  ITensorProxyPtr tensor = (*slice)->getOutput(0);

  // Reshape for shrink axis, ellipsis masks based on the shape computed by
  // ValidateStridedSliceOp or HandleDynamicStridedSliceInput.
  nvinfer1::Dims dims = tensor->trt_tensor()->getDimensions();
  std::vector<int> slice_input_dims(dims.d, dims.d + dims.nbDims);
  StridedSliceShapeSpec empty_spec;
  empty_spec.shrink_axis_dense_mask = 0;
  auto shrink_axis_mask =
      strided_slice_spec.value_or(empty_spec).shrink_axis_dense_mask;
  if (final_shape) {
    if (shrink_axis_mask) {
      int shrink_idx = params->use_implicit_batch ? 1 : 0;
      const auto bShrink_axis_mask = std::bitset<32>(shrink_axis_mask);
      for (int idx = 0; idx < slice_input_dims.size(); ++idx, ++shrink_idx) {
        const bool shrink_axis = bShrink_axis_mask[shrink_idx];
        if (shrink_axis) {
          slice_input_dims[idx] = 0;
        }
      }
<<<<<<< refs/remotes/origin/master
      TF_RETURN_IF_ERROR(params->converter->SqueezeTensor(
          tensor, &slice_input_dims, params, &tensor, op_instance));
    } else {
      /* To do: pmajety:
            Remove the else condition when shrink_axis_mask is always defined */
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params->converter, TRT_TensorOrWeights(tensor), *final_shape,
          /*validation_only=*/false, &tensor, node_def, op_instance));
=======
      TF_RETURN_IF_ERROR(params->converter->SqueezeTensor(tensor,&slice_input_dims,
                      params, &tensor, op_instance));
    }
    else {
    /* To do: pmajety:
          Remove the else condition when shrink_axis_mask is always defined. */
        TF_RETURN_IF_ERROR(PrepareTensorForShape(
                params->converter, TRT_TensorOrWeights(tensor), *final_shape,
                /*validation_only=*/false, &tensor, node_def, op_instance));
>>>>>>> Negative values in begin/end tensors -- preliminary changes.
    }
  }
  params->outputs->push_back(TRT_TensorOrWeights(tensor));
  return OkStatus();
}
Status ProcessOutOfRangeBeginAndEnd(
    TRTNetworkBuilder* builder, 
    const StridedSliceShapeSpec& strided_slice_spec,
    const absl::InlinedVector<int64, 4>& process_begin_indices,
    const absl::InlinedVector<int64, 4>& process_end_indices,
    nvinfer1::Dims begin_dims, nvinfer1::Dims stride_dims,
    nvinfer1::Dims end_dims){

  // Vectors for analyzing negative begin and end endices.
  absl::InlinedVector<int64, 4> negative_begin_indices;
  absl::InlinedVector<int64, 4> negative_end_indices;

  VLOG(2) << "begin_dims before: " <<  DebugString(begin_dims);
  VLOG(2) << "end_dims before: " <<  DebugString(end_dims);
  const auto begin_mask = std::bitset<32>(strided_slice_spec.begin_dense_mask);
  const auto end_mask = std::bitset<32>(strided_slice_spec.end_dense_mask);
  const auto shrink_axis_mask = std::bitset<32>(strided_slice_spec.shrink_axis_dense_mask);
  nvinfer1::Dims dims = input_tensor->getDimensions();


  for (int i = 0; i < process_begin_indices.size(); i++) {
    auto idx = dynamic_input_size_indices[i];
    if (!begin_mask[idx] && begin_dims.d[idx] < 0) {
        negative_begin_indices.push_back(idx);
    }
  }
  for (int i = 0; i < process_end_indices.size(); i++) {
    if (!end_mask[idx] && end_dims.d[idx] < 0) {
        negative_end_indices.push_back(idx);
    }
  }

  VLOG(2) << " Negative begin indices: " << DebugString(negative_begin_indices)
          << " Negative end indices: " << DebugString(negative_end_indices);
  // Make the negative indices in begin and end tensors positive and in range
  // Eg: shape : [3, 4, 2]
  //     begin : [-1, 2, 0]  should be transformed to [2, 2, 0]
  if(negative_begin_indices.size() > 0){
    // return shape-1 tensor
    StatusOr<nvinfer1::IGatherLayer*> negIdx_shape_masked_tensor =
        builder->GetPartialShapeOf(input_tensor, negative_begin_indices,
                                   /*sub_one=*/true);
    TRT_ENSURE_PTR_OK(negIdx_shape_masked_tensor); // [2, 0, 0] 
    // inverse mask
    std::vector<int> non_neg_begin;
    for(int idx = 0; idx < begin_dims.nbDims; idx++){
      if(neg_beg_set.find(idx) != neg_beg_set.end()){
        // is a negative index, so mask should mark the value 0.
        non_neg_begin[idx] = 0;
      }
      else non_neg_begin[idx] = begin_dims.d[idx];

    }
    VLOG(2) << "Begin tensor: "<< DebugString(begin_tensor);
    StatusOr<nvinfer1::IConstantLayer*> nonNegIdx_begin_masked_tensor =
        builder->Constant(non_neg_begin);        //[0,2,0] 
    TRT_ENSURE_PTR_OK(nonNegIdx_begin_masked_tensor);
 
     // Add back the original "begin" values for non negative dimensions
     // with inverse mask generated values. 
   StatusOr<nvinfer1::IElementWiseLayer*> begin_intermediate
        = builder->Add((*negIdx_shape_masked_tensor)->getOutput(0),
                       (*nonNegIdx_begin_masked_tensor)->getOutput(0)); //[1,2,0]
   TRT_ENSURE_PTR_OK(begin_intermediate);
     //begin_tensor = (*begin_intermediate)->getOutput(0);
 
    // If the begin value at a dimension is less than -1 * input size,
    // then the begin value needs to be reset to 0.
     auto full_runtime_input_shape = builder->Shape(input_tensor);
     TRT_ENSURE_PTR_OK(full_runtime_input_shape); //[3,4,2]
    // -1 * shape layer
    // layer of negative ones
    auto neg_ones_layer = builder->Constant(std::vector<int>(begin_dims.nbDims,-1));
    TRT_ENSURE_PTR_OK(neg_ones_layer);//[-1,-1,-1]
    
    auto neg_runtime_input_shape = builder->Mul(
        (*neg_ones_layer)->getOutput(0), (*shape_layer)->getOutput(0)); //[-3, -4, -2]
    TRT_ENSURE_PTR_OK(neg_runtime_input_shape);

    StatusOr<nvinfer1::IElementWiseLayer*> out_of_range_dims_begin_left_bool =
        builder->Greater(begin_tensor, (*neg_runtime_input_shape)->getOutput(0));
    TRT_ENSURE_PTR_OK(out_of_range_dims_begin_left_bool); //[1,1,1]
    //(*out_of_range_dims_begin_left_bool)->setOutputType(0, nvinfer1::DataType::kINT32); 
    auto out_of_range_dims_begin_left_int32 = builder->ChangeOutputType(
                   out_of_range_dims_begin_left_bool,
                   nvinfer1::DataType::kINT32);

    StatusOr<nvinfer1::IElementWiseLayer*> neg_begin_corrected =
        builder->Mul((*begin_intermediate)->getOutput(0), (*out_of_range_dims_begin_left_int32)->getOutput(0));
       //[2, 2, 0] 
    TRT_ENSURE_PTR_OK(neg_begin_corrected);
    begin_tensor = (*neg_begin_corrected)->getOutput(0);

    }



    }
Status HandleDynamicStridedSliceInput(
    TRTNetworkBuilder* builder, nvinfer1::ISliceLayer* slice_layer,
    const StridedSliceShapeSpec& strided_slice_spec,
    const absl::InlinedVector<int64, 4>& dynamic_input_size_indices,
    nvinfer1::Dims begin_dims, nvinfer1::Dims stride_dims,
    nvinfer1::Dims end_dims) {
  TRT_ENSURE(builder);
  TRT_ENSURE(slice_layer);

  nvinfer1::ITensor* input_tensor = slice_layer->getInput(0);
  TRT_ENSURE(input_tensor);

  // When begin_mask or end_mask are set, we have to disregard the begin_tensor
  // and end_tensor values. In static indices cases, ValidateStridedSliceOp
  // returns the correct begin_tensor and end_tensor values, however with
  // dynamic indices the correct shape has to be computed.

  VLOG(2) << "begin_dims before: " <<  DebugString(begin_dims);
  VLOG(2) << "end_dims before: " <<  DebugString(end_dims);
  const auto begin_mask = std::bitset<32>(strided_slice_spec.begin_dense_mask);
  const auto end_mask = std::bitset<32>(strided_slice_spec.end_dense_mask);
  const auto shrink_axis_mask = std::bitset<32>(strided_slice_spec.shrink_axis_dense_mask);
  nvinfer1::Dims dims = input_tensor->getDimensions();

  for (int idx = 0; idx < dims.nbDims; ++idx){
    // VLOG(2) << "begin_mask[" << idx << "]: "<< begin_mask[idx];
    // VLOG(2) << "end_mask[" << idx << "]: "<< end_mask[idx];
    // VLOG(2) << "shrink_mask[" << idx << "]: "<< shrink_axis_mask[idx];
    if (begin_mask[idx]) {
      begin_dims.d[idx] = 0;
    }
    if (end_mask[idx] && !shrink_axis_mask[idx]){
      end_dims.d[idx] = dims.d[idx];
    }
    if(shrink_axis_mask[idx]){
       end_dims.d[idx] = begin_dims.d[idx] + 1;
    }
  }

  VLOG(2) << "begin_dims after shrink_axis_mask correction: " <<  DebugString(begin_dims);
  VLOG(2) << "end_dims after shrink_axis_mask correction: " <<  DebugString(end_dims);

  // For each dynamic input dimension of the input, do some preprocessing based
  // on whether this dimension is set in "begin_mask" or "end_mask" and the sign
  // of the dimension's stride value.
  // When stride is negative:
  //   - If "begin_mask[dynamic_idx]" is set, then we need to adjust the slice
  //     start of dimension[i] to the dynamic size.
  //   - If "end_mask[dynamic_idx]" is set, it suffices to set
  //     end_dims[dynamic_idx] to -1.
  // When stride is positive:
  //   - If "begin_mask[dynamic_idx]" is set, it suffices to set
  //     begin_dims[dynamic_idx] to zero.
  //   - If "end_mask[dynamic_idx]" is set, we need to adjust slice end to the
  //     dynamic size of dimension "dynamic_idx".
  absl::InlinedVector<int64, 4> dynamic_begin_indices;
  absl::InlinedVector<int64, 4> dynamic_end_indices;

  for (int i = 0; i < dynamic_input_size_indices.size(); i++) {
    auto dynamic_idx = dynamic_input_size_indices[i];
    if (begin_dims.d[dynamic_idx] > -1 && begin_mask[dynamic_idx]) {
      begin_dims.d[dynamic_idx] = 0;
      if (stride_dims.d[dynamic_idx] < 0) {
        dynamic_begin_indices.push_back(dynamic_idx);
      }
    }
<<<<<<< refs/remotes/origin/master
    if (end_mask[dynamic_idx] && !shrink_axis_mask[dynamic_idx]) {
=======
    if (end_dims.d[dynamic_idx] > -1 && end_mask[dynamic_idx] && !shrink_axis_mask[dynamic_idx]) {
      end_dims.d[dynamic_idx] = stride_dims.d[dynamic_idx] > 0 ? 0 : -1;
>>>>>>> Negative values in begin/end tensors -- preliminary changes.
      if (stride_dims.d[dynamic_idx] > 0) {
        dynamic_end_indices.push_back(dynamic_idx);
    }
  }

  VLOG(2) << " Dynamic begin indices: " << DebugString(dynamic_begin_indices)
          << " Dynamic end indices: " << DebugString(dynamic_end_indices);

  // Create ITensors for each of the begin/stride/end constants.
  StatusOr<nvinfer1::IConstantLayer*> begin_const = builder->Constant(
      std::vector<int>(begin_dims.d, begin_dims.d + begin_dims.nbDims));
  TRT_ENSURE_PTR_OK(begin_const);
  nvinfer1::ITensor* begin_tensor = (*begin_const)->getOutput(0);
  StatusOr<nvinfer1::IConstantLayer*> stride_const = builder->Constant(
      std::vector<int>(stride_dims.d, stride_dims.d + stride_dims.nbDims));
  TRT_ENSURE_PTR_OK(stride_const);
  StatusOr<nvinfer1::IConstantLayer*> end_const = builder->Constant(
      std::vector<int>(end_dims.d, end_dims.d + end_dims.nbDims));
  TRT_ENSURE_PTR_OK(end_const);
  nvinfer1::ITensor* end_tensor = (*end_const)->getOutput(0);
  std::set<int> neg_beg_set;
  for(int idx : negative_begin_indices){
    neg_beg_set.insert(idx);
  }

  // Make the negative indices in begin and end tensors positive and in range
  // Eg: shape : [3, 4, 2]
  //     begin : [-1, 2, 0]  should be transformed to [2, 2, 0]
  if(negative_begin_indices.size() > 0){
    // return shape-1 tensor
    StatusOr<nvinfer1::IGatherLayer*> negIdx_shape_masked_tensor =
        builder->GetPartialShapeOf(input_tensor, negative_begin_indices,
                                   /*sub_one=*/true);
    TRT_ENSURE_PTR_OK(negIdx_shape_masked_tensor); // [2, 0, 0] 
//    // inverse mask
    std::vector<int> non_neg_begin;
    for(int idx = 0; idx < begin_dims.nbDims; idx++){
      if(neg_beg_set.find(idx) != neg_beg_set.end()){
        // is a negative index, so mask should mark the value 0.
        non_neg_begin[idx] = 0;
      }
      else non_neg_begin[idx] = begin_dims.d[idx];

    }
    VLOG(2) << "Begin tensor: "<< DebugString(begin_tensor);
    StatusOr<nvinfer1::IConstantLayer*> nonNegIdx_begin_masked_tensor =
        builder->Constant(non_neg_begin);        //[0,2,0] 
    TRT_ENSURE_PTR_OK(nonNegIdx_begin_masked_tensor);
//
//    // Add back the original "begin" values for non negative dimensions
//    // with inverse mask generated values. 
    StatusOr<nvinfer1::IElementWiseLayer*> begin_intermediate
         = builder->Add((*negIdx_shape_masked_tensor)->getOutput(0),
                        (*nonNegIdx_begin_masked_tensor)->getOutput(0)); //[1,2,0]
    TRT_ENSURE_PTR_OK(begin_intermediate);
//    //begin_tensor = (*begin_intermediate)->getOutput(0);
//
//   // If the begin value at a dimension is less than -1 * input size,
//   // then the begin value needs to be reset to 0.
     auto full_runtime_input_shape = builder->Shape(input_tensor);
     TRT_ENSURE_PTR_OK(full_runtime_input_shape); //[3,4,2]
    // -1 * shape layer
    // layer of negative ones
    auto neg_ones_layer = builder->Constant(std::vector<int>(begin_dims.nbDims,-1));
    TRT_ENSURE_PTR_OK(neg_ones_layer);//[-1,-1,-1]
    
    auto neg_runtime_input_shape = builder->Mul(
        (*neg_ones_layer)->getOutput(0), (*shape_layer)->getOutput(0)); //[-3, -4, -2]
    TRT_ENSURE_PTR_OK(neg_runtime_input_shape);

    StatusOr<nvinfer1::IElementWiseLayer*> out_of_range_dims_begin_left_bool =
        builder->Greater(begin_tensor, (*neg_runtime_input_shape)->getOutput(0));
    TRT_ENSURE_PTR_OK(out_of_range_dims_begin_left_bool); //[1,1,1]
    //(*out_of_range_dims_begin_left_bool)->setOutputType(0, nvinfer1::DataType::kINT32); 
    auto out_of_range_dims_begin_left_int32 = builder->ChangeOutputType(
                   out_of_range_dims_begin_left_bool,
                   nvinfer1::DataType::kINT32);

    StatusOr<nvinfer1::IElementWiseLayer*> neg_begin_corrected =
        builder->Mul((*begin_intermediate)->getOutput(0), (*out_of_range_dims_begin_left_int32)->getOutput(0));
       //[2, 2, 0] 
    TRT_ENSURE_PTR_OK(neg_begin_corrected);
    begin_tensor = (*neg_begin_corrected)->getOutput(0);

  }
  // Make corrections based on the begin_mask/end_mask values.
  if (dynamic_end_indices.size() > 0) {
    StatusOr<nvinfer1::IGatherLayer*> dynamic_end_masked_tensor =
        builder->GetPartialShapeOf(input_tensor, dynamic_end_indices,
                                   /*sub_one=*/false);
    TRT_ENSURE_PTR_OK(dynamic_end_masked_tensor);
    StatusOr<nvinfer1::IElementWiseLayer*> end_corrected =
        builder->Add((*dynamic_end_masked_tensor)->getOutput(0), end_tensor);
    TRT_ENSURE_PTR_OK(end_corrected);
    end_tensor = (*end_corrected)->getOutput(0);
  }
  if (dynamic_begin_indices.size() > 0) {
    StatusOr<nvinfer1::IGatherLayer*> dynamic_begin_shape_masked_tensor =
        builder->GetPartialShapeOf(input_tensor, dynamic_begin_indices,
                                   /*sub_one=*/true);
    TRT_ENSURE_PTR_OK(dynamic_begin_shape_masked_tensor);

    // Add back the original "begin" values for static dimensions.
    StatusOr<nvinfer1::IElementWiseLayer*> begin_corrected = builder->Add(
        (*dynamic_begin_shape_masked_tensor)->getOutput(0), begin_tensor);
    TRT_ENSURE_PTR_OK(begin_corrected);
    begin_tensor = (*begin_corrected)->getOutput(0);
  }

  // Calculate the final size of the slice dynamicaly.
  nvinfer1::ITensor* size_tensor;
  {
    StatusOr<nvinfer1::IElementWiseLayer*> num =
        builder->Sub(end_tensor, begin_tensor);
    TRT_ENSURE_PTR_OK(num);
    StatusOr<nvinfer1::IElementWiseLayer*> ceil_div = builder->AbsCeilDivInt(
        (*num)->getOutput(0), (*stride_const)->getOutput(0));
    TRT_ENSURE_PTR_OK(ceil_div);
    size_tensor = (*ceil_div)->getOutput(0);
  }

  slice_layer->setInput(1, *begin_tensor);
  slice_layer->setInput(2, *size_tensor);
  slice_layer->setInput(3, *(*stride_const)->getOutput(0));

  return OkStatus();
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
