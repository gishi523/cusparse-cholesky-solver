#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include <cuda_runtime.h>

#include "macro.h"

template <typename T>
struct DeviceBuffer
{
	DeviceBuffer() : data(nullptr), size(0) {}
	DeviceBuffer(size_t size) : data(nullptr), size(0) { allocate(size); }
	~DeviceBuffer() { destroy(); }

	void allocate(size_t _size)
	{
		if (data && size >= _size)
			return;

		destroy();
		CUDA_CHECK(cudaMalloc(&data, sizeof(T) * _size));
		size = _size;
	}

	void destroy()
	{
		if (data)
			CUDA_CHECK(cudaFree(data));
		data = nullptr;
		size = 0;
	}

	void upload(const T* h_data)
	{
		CUDA_CHECK(cudaMemcpy(data, h_data, sizeof(T) * size, cudaMemcpyHostToDevice));
	}

	void download(T* h_data)
	{
		CUDA_CHECK(cudaMemcpy(h_data, data, sizeof(T) * size, cudaMemcpyDeviceToHost));
	}

	void copyTo(DeviceBuffer& rhs) const
	{
		CUDA_CHECK(cudaMemcpy(rhs.data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
	}

	void copyTo(T* rhs) const
	{
		CUDA_CHECK(cudaMemcpy(rhs, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
	}

	void fillZero()
	{
		CUDA_CHECK(cudaMemset(data, 0, sizeof(T) * size));
	}

	bool empty() const { return !(data && size > 0); }

	void assign(size_t size, const T* h_data)
	{
		allocate(size);
		upload(h_data);
	}

	T* data;
	size_t size;
};

#endif // !__DEVICE_BUFFER_H__
