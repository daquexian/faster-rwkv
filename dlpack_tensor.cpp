#include "dlpack_tensor.h"

namespace rwkv {

Device ConvertToRwkvDevice(const ::DLDevice ctx) {
  switch (ctx.device_type)
  {
  case DLDeviceType::kDLCPU:
      return Device::kCPU;
  case DLDeviceType::kDLCUDA:
      return Device::kCUDA;
  default:
      RV_UNIMPLEMENTED() << "Unsupported device type: " << ctx.device_type;
  }
}

rwkv::DType ConvertToRwkvDataType(const ::DLDataType &dtype) {
  DType rwkvdtype = DType::kUndefined;
  switch (dtype.code) {
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 4: rwkvdtype = DType::kInt4; break;
        case 8: rwkvdtype = DType::kInt8; break;
        case 32: rwkvdtype = DType::kInt32; break;
        case 64: rwkvdtype = DType::kInt64; break;
        default:
          RV_UNIMPLEMENTED() << "Unsupported data type: " << dtype.code << dtype.bits;
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16: rwkvdtype = DType::kFloat16; break;
        case 32: rwkvdtype = DType::kFloat32; break;
        default:
          RV_UNIMPLEMENTED() << "Unsupported data type: " << dtype.code << dtype.bits;
      }
      break;
    default: RV_UNIMPLEMENTED() << "Unsupported code " << dtype.code;
  }
  RV_CHECK(rwkvdtype != DType::kUndefined);
  return rwkvdtype;
}

Tensor fromDLPack(const ::DLManagedTensor *src) {
  const auto& dl_tensor = src->dl_tensor;
  Device device = ConvertToRwkvDevice(dl_tensor.device);
  DType dtype = ConvertToRwkvDataType(dl_tensor.dtype);

  // Build Tensor
  const Shape shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);

  size_t array_size_in_bytes = shape.size() * elem_size(dtype);
  Tensor rwkv_tensor = Tensor::FromPtr(dl_tensor.data, shape, dtype, device);
  return rwkv_tensor;
}

::DLDevice ConvertToDLDevice(const rwkv::Device &device) {
  ::DLDevice dl_device;
  switch (device){
      case Device::kCPU:
          dl_device.device_type = ::DLDeviceType::kDLCPU;
          dl_device.device_id = 0;
          break;
      case Device::kCUDA:
          dl_device.device_type = ::DLDeviceType::kDLCUDA;
          dl_device.device_id = 1;
          break;
      default:
          RV_UNIMPLEMENTED() << "Unsupport device type: " << static_cast<int>(device);
  }
  return dl_device;
}

::DLDataType ConvertToDLDataType(const rwkv::DType &dtype) {
  ::DLDataType dl_dtype;
  dl_dtype.lanes = 1;
  dl_dtype.bits = rwkv::elem_size(dtype) * 8;
  switch (dtype) {
    case DType::kInt8:
        dl_dtype.code = DLDataTypeCode::kDLInt;
        break;
    case DType::kInt32:
        dl_dtype.code = DLDataTypeCode::kDLInt;
        break;
    case DType::kInt64:
        dl_dtype.code = DLDataTypeCode::kDLInt;
        break;
    case DType::kFloat16:
        dl_dtype.code = DLDataTypeCode::kDLFloat;
        break;
    case DType::kFloat32:
        dl_dtype.code = DLDataTypeCode::kDLFloat;
        break;    
    default:
        RV_UNIMPLEMENTED() << "Unsupport data type: " << static_cast<int>(dtype);
        break;
  }
  return dl_dtype;
}

struct ATenDLMTensor {
  Tensor handle = Tensor::Empty({2, 2} /* shape */, DType::kFloat32, Device::kCPU);
  DLManagedTensor tensor;
};

static void deleter(DLManagedTensor* arg) {
  delete static_cast<ATenDLMTensor*>(arg->manager_ctx);
}

::DLManagedTensor* toDLPack(const Tensor& src) {
  ATenDLMTensor* atDLMTensor(new ATenDLMTensor);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter;
  atDLMTensor->tensor.dl_tensor.data = const_cast<void *>(src.data_ptr());
  atDLMTensor->tensor.dl_tensor.device = ConvertToDLDevice(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = src.ndim();
  atDLMTensor->tensor.dl_tensor.dtype = ConvertToDLDataType(src.dtype());
  atDLMTensor->tensor.dl_tensor.shape =
      const_cast<int64_t*>(src.shape().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  return &(atDLMTensor->tensor);
}
}