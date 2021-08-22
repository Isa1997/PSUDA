
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "curand.h"
#include "curand_kernel.h"
#include <iostream>
#include <limits>
#include "../PSUDA/Test.h"

# define DELLEXPORT extern "C" __declspec(dllexport)

#define W (0.75f)
#define C1 (1.f)
#define C2 (2.f)

#define SIZEFLOAT (sizeof(float))
#define SIZEINT (sizeof(int))
#define SIZEUINT (sizeof(unsigned))

__device__
float Fitness(float x)
{
    return FitnessFunction(x);
}

__device__
int Randomize(float seed, int threadId)
{
    return static_cast<int>(seed * 100) + threadId;
}

__global__  void Init(float* positions,float* velocities, float* fitnesses, float* bestPositions, float randomSeed)
{
    curandState state;
    curand_init(Randomize(randomSeed, threadIdx.x), 0, 1, &state);
    float position = curand_uniform(&state) * 40.f - 20.f ;
    float velocity = curand_uniform(&state) * 2.f - 1.f;
    float currentValue = Fitness(position);
    float localBest = currentValue;
    float positionBest = position;

    positions[threadIdx.x] = position;
    velocities[threadIdx.x] = velocity;
    fitnesses[threadIdx.x] = localBest;
    bestPositions[threadIdx.x] = positionBest;
    __syncthreads();
}

__global__  void UpdateGlobalValues(unsigned swarmSize, float* fitnesses, float* bestPositions, float* bestGlobalValue, float* BestGlobalPosition)
{
    *bestGlobalValue = LONG_MAX;
    for (int i = 0; i < swarmSize; ++i)
    {
        if (fitnesses[i] < *bestGlobalValue)
        {
            *bestGlobalValue = fitnesses[i];
            *BestGlobalPosition = bestPositions[i];
        }
    }
    __syncthreads();
}

__global__  void UpdatePositionAndVelocity(float* positions, float* velocities, float* fitnesses, float* bestPositions, float* bestGlobalPosition, float r1, float r2)
{
    velocities[threadIdx.x] = W * velocities[threadIdx.x] + C1 * r1 * (bestPositions[threadIdx.x]- positions[threadIdx.x]) 
        + C2 * r2 * (*bestGlobalPosition - positions[threadIdx.x]);
    positions[threadIdx.x] += velocities[threadIdx.x];

    if (Fitness(positions[threadIdx.x]) < Fitness(bestPositions[threadIdx.x]))
    {
        bestPositions[threadIdx.x] = positions[threadIdx.x];
        fitnesses[threadIdx.x] = Fitness(positions[threadIdx.x]);
    }
    __syncthreads();
}

DELLEXPORT void RingPSO(unsigned iterations, unsigned swarmSize, float r1, float r2, float* max)
{
    //Init for GPU vars
    float* bestGlobalValue;
    float* bestGlobalPosition;

    //Arrars
    float* fitness;
    float* bestPositions;
    float* positions;
    float* velocities;

    cudaMalloc((void**)&bestGlobalValue, SIZEFLOAT);
    cudaMalloc((void**)&bestGlobalPosition, SIZEFLOAT);
    cudaMalloc((void**)&bestPositions, SIZEFLOAT * swarmSize);
    cudaMalloc((void**)&positions, SIZEFLOAT * swarmSize);
    cudaMalloc((void**)&fitness, SIZEFLOAT * swarmSize);
    cudaMalloc((void**)&velocities, SIZEFLOAT * swarmSize);

    //Run kernel function
    Init<<<1, swarmSize >>> (positions, velocities, fitness, bestPositions, r1);

    UpdateGlobalValues <<<1, 1 >>> (swarmSize, fitness, bestPositions, bestGlobalValue, bestGlobalPosition);

    for (int i = 0; i < iterations; ++i)
    {
        UpdatePositionAndVelocity <<<1, swarmSize >>> (positions, velocities, fitness, bestPositions, bestGlobalPosition, r1, r2);
        UpdateGlobalValues<<<1,1>>> (swarmSize, fitness, bestPositions, bestGlobalValue, bestGlobalPosition);
    }

    //Copy into main memory from GPU
    cudaMemcpy(max, bestGlobalValue, SIZEFLOAT, cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(bestPositions);
    cudaFree(bestGlobalPosition);
    cudaFree(bestGlobalValue);
    cudaFree(positions);
    cudaFree(velocities);
    cudaFree(fitness); 
}
