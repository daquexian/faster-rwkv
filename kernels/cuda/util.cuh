#ifndef _UTIL_CUH_
#define _UTIL_CUH_

#include <stdint.h>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cub/cub.cuh>

#include <nvfunctional>

#include <stdio.h>

#include "elementwise.cuh"

namespace cuda
{
	namespace kernel
	{
		template<typename T>
		__global__ void transpose(T* from, T* to, uint32_t width, uint32_t height)
		{
		    __shared__ T block[16][16 + 1];
		
		    uint32_t x_index = blockIdx.x * 16 + threadIdx.x;
		    uint32_t y_index = blockIdx.y * 16 + threadIdx.y;
		
		    if ((x_index < width) && (y_index < height))
		    {
		        uint32_t index_in = y_index * width + x_index;
		        block[threadIdx.y][threadIdx.x] = from[index_in];
		    }
		
		    __syncthreads();
		
		    x_index = blockIdx.y * 16 + threadIdx.x;
		    y_index = blockIdx.x * 16 + threadIdx.y;
		
		    if ((x_index < height) && (y_index < width))
		    {
		        uint32_t index_out = y_index * height + x_index;
		        to[index_out] = block[threadIdx.x][threadIdx.y];
		    }
		}
	}

	template<typename T_FROM, typename T_TO>
	inline cudaError_t convert(T_FROM* from, T_TO* to, uint32_t count)
	{
		using half2float_t = elementwise::functor::cast::functor_t<half, float>;
		using float2half_t = elementwise::functor::cast::functor_t<float, half>;

		if constexpr (sizeof(T_FROM) == 2 && sizeof(T_TO) == 4)
			elementwise::unary(half2float_t(), count, (float*)to, (half*)from);

		else if (sizeof(T_FROM) == 4 && sizeof(T_TO) == 2)
			elementwise::unary(float2half_t(), count, (half*)to, (float*)from);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t transpose(const uint32_t m, const uint32_t k, T* from, T* to)
	{
		dim3 grid((k + 15) / 16, (m + 15) / 16, 1);
		dim3 threads(16, 16, 1);

		if constexpr (sizeof(T) == 2)
			kernel::transpose<half><<<grid, threads>>> ((half*)from, (half*)to, k, m);
		else
			kernel::transpose<float><<<grid, threads>>> ((float*)from, (float*)to, k, m);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_product(const uint32_t count, T* a, T* b, T* c)
	{
		using product_t = elementwise::functor::product::functor_t<T, T>;

		elementwise::binary(product_t(), count, c, a, b);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_add(const uint32_t count, T* a, T* b, T* c)
	{
		using add_t = elementwise::functor::add::functor_t<T, T>;

		elementwise::binary(add_t(), count, c, a, b);

		return cudaGetLastError();
	}

	inline cudaError_t dump_fp16(float* h_dst, half* d_src, uint32_t count)
	{
		float* fp32 = 0;

		cudaMalloc(&fp32, sizeof(float) * count);

		convert(d_src, fp32, count);

		cudaMemcpy(h_dst, fp32, sizeof(float) * count, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaFree(fp32);

		return cudaGetLastError();
	}
}


#endif

