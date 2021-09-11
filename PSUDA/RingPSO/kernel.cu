
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
float Fitness(float* x,const short dimension)
{
    return FitnessFunction(x, dimension);
}

__device__
int Randomize(float seed, int threadId)
{
    return static_cast<int>(seed * 100) + threadId;
}

__global__  void Init(float** positions,float** velocities, float* fitnesses, float** bestPositions, float randomSeed, const short dimension)
{
    // Position Init
    float* position = (float*)malloc(SIZEFLOAT * dimension);

    for (short i = 0; i < dimension; ++i)
    {
        curandState state;
        curand_init(Randomize(randomSeed, threadIdx.x), 0, 1, &state);
        position[i] = curand_uniform(&state) * 40.f - 20.f;
    }

    //Velocity Init
    float* velocity = (float*)malloc(SIZEFLOAT * dimension);
    for (short i = 0; i < dimension; ++i)
    {
        curandState state;
        curand_init(Randomize(randomSeed, threadIdx.x), 0, 1, &state);
        velocity[i] = curand_uniform(&state) * 2.f - 1.f;
    }

    float currentValue = Fitness(position,dimension);
    float localBest = currentValue;
    float* positionBest = position;

    positions[threadIdx.x] = position;
    velocities[threadIdx.x] = velocity;
    fitnesses[threadIdx.x] = localBest;

    float* row = (float*)((char*)bestPositions + threadIdx.x * sizeof(float) * dimension);
    for (short j = 0; j < dimension; ++j)
    {
        row[j] = positionBest[j];
    }
    __syncthreads();
}

__global__  void UpdateGlobalValues(unsigned swarmSize, const short dimension, float* fitnesses, float** bestPositions, float* bestGlobalValue, float* bestGlobalPosition)
{
    *bestGlobalValue = LONG_MAX;
    for (int i = 0; i < swarmSize; ++i)
    {
        if (fitnesses[i] < *bestGlobalValue)
        {
            *bestGlobalValue = fitnesses[i];
            float* row = (float*)((char*)bestPositions + i * sizeof(float) * dimension);
            for (short j = 0; j < dimension; ++j)
            {
                bestGlobalPosition[j] = row[j];
            }
            
        }
    }
    __syncthreads();
}

__global__  void UpdatePositionAndVelocity(float** positions, float** velocities, float* fitnesses, float** bestPositions, float* bestGlobalPosition, float r1, float r2, const short dimension)
{

    float* velocityToUpdate = velocities[threadIdx.x];
    float* row = (float*)((char*)bestPositions + threadIdx.x * sizeof(float) * dimension);
    for (int i = 0; i < dimension; ++i)
    {
        velocityToUpdate[i] = W* velocityToUpdate[i] + C1 * r1 * (row[i] - positions[threadIdx.x][i])
            + C2 * r2 * (*bestGlobalPosition - positions[threadIdx.x][i]);
    }
    for (int i = 0; i < dimension; ++i)
    {
        positions[threadIdx.x][i] += velocities[threadIdx.x][i];
    }

    float currentFitness = Fitness(positions[threadIdx.x], dimension);
    if ( currentFitness < fitnesses[threadIdx.x])
    {
        fitnesses[threadIdx.x] = currentFitness;
 
        float* row = (float*)((char*)bestPositions + threadIdx.x * sizeof(float) * dimension);
        for (short j = 0; j < dimension; ++j)
        {
            row[j] = positions[threadIdx.x][j];
        }
    }
    __syncthreads();
}

DELLEXPORT void RingPSO(unsigned iterations, unsigned swarmSize, float r1, float r2, float* max, float* bestPosition, const short dimension)
{
    if (dimension < 0)
    {
        std::cerr << "Size of position vector is negative!\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    else if (dimension > 3)
    {
        std::cout << "Case is not suported!(yet :D)\n" << std::endl;
        exit(EXIT_SUCCESS);
    }

    //Init for GPU vars
    float* bestGlobalValue;
    float* bestGlobalPosition;

    //Arrars
    float* fitness;
    float** bestPositions;
    float** positions;
    float** velocities;

    size_t pitch;
    cudaMallocPitch((void**)&bestPositions, &pitch, SIZEFLOAT * dimension, swarmSize);
    cudaMalloc((void**)&bestGlobalValue, SIZEFLOAT);
    cudaMalloc((void**)&bestGlobalPosition, SIZEFLOAT * dimension);
    cudaMalloc((void**)&positions, sizeof(float*) * swarmSize);
    cudaMalloc((void**)&fitness, SIZEFLOAT * swarmSize);
    cudaMalloc((void**)&velocities, sizeof(float*) * swarmSize);

    //Run kernel function
    Init<<<1, swarmSize >>> (positions, velocities, fitness, bestPositions, r1, dimension);

    UpdateGlobalValues <<<1, 1 >>> (swarmSize, dimension, fitness, bestPositions, bestGlobalValue, bestGlobalPosition);

    for (int i = 0; i < iterations; ++i)
    {
        UpdatePositionAndVelocity <<<1,swarmSize >>> (positions, velocities, fitness, bestPositions, bestGlobalPosition, r1, r2, dimension);
        UpdateGlobalValues<<<1,1>>> (swarmSize, dimension, fitness, bestPositions, bestGlobalValue, bestGlobalPosition);
    }

    //Copy into main memory from GPU
    cudaMemcpy(max, bestGlobalValue, SIZEFLOAT, cudaMemcpyDeviceToHost);
    cudaMemcpy(bestPosition, bestGlobalPosition, SIZEFLOAT * dimension, cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(bestPositions);
    cudaFree(bestGlobalPosition);
    cudaFree(bestGlobalValue);
    cudaFree(positions);
    cudaFree(velocities);
    cudaFree(fitness); 
}
