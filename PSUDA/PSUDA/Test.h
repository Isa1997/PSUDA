#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"

#define DELLEXPORT extern "C" __declspec(dllexport)

DELLEXPORT __host__ __device__ float FitnessFunction(float x)
{
	return x * x + 17.1435;
   // return x * x * x * x - 6 * x * x + 4 * x + 11;
}