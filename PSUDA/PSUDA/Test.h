#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include <cstdlib>

#define boundLeft (-10)
#define boundRight (10)
#define between(x,a,b) ((x>a) && (x<b))
#define DELLEXPORT extern "C" __declspec(dllexport)

DELLEXPORT __host__ __device__ float FitnessFunction(float* x, const short size)
{
	if (size == 1)
	{
		return x[0] * x[0] - 5 * x[0] + 6;
	}
	else if (size == 2)
	{
		//if (between(x[0], boundLeft, boundRight) && between(x[1], boundLeft, boundRight))
		{
			float d = -std::fabs(std::sin(x[0])*std::cos(x[1])*std::exp(std::fabs(1-(std::sqrt(x[0]*x[0] + x[1]*x[1])/3.1415))));
			return d;
		}
		// (boundRight + 2 * boundRight - 7)* (boundRight + 2 * boundRight - 7) + (2 * boundRight + boundRight - 5) * (2 * boundRight + boundRight - 5);
	}
	else if (size == 3)
	{
		return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + 3.0f;
	}
	return 0;
}