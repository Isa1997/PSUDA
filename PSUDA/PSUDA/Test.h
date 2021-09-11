#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"

#define DELLEXPORT extern "C" __declspec(dllexport)

DELLEXPORT __host__ __device__ float FitnessFunction(float* x, const short size)
{
	if (size == 1)
	{
		return x[0] * x[0] + 5 * x[0] + 6;
		//return	x[0] * x[0] - 17.1435f;;
	}
	else if (size == 2)
	{
		return  x[0] * x[0] + x[1] * x[1] + 2;
	}
	else if (size == 3)
	{
		return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + 3.0f;
	}
	return 0;
}

/*
Some test function: x * x * x * x - 6 * x * x + 4 * x + 11;
*/