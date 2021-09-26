
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand.h"
#include "curand_kernel.h"
#include <limits>
#include <iostream>
#include "../PSUDA/Test.h"

# define DELLEXPORT extern "C" __declspec(dllexport)

#define W (0.7298f)
#define C1 (1.49f)
#define C2 (1.49f)

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

__global__  void Init(float** positions,float** velocities, float* fitnesses, float** bestPositions, float* bestGlobalValue, const  float randomSeed, const short dimension)
{
    // Position Init
    int localId = threadIdx.x;
    float* position = (float*)malloc(SIZEFLOAT * dimension);
    float* velocity = (float*)malloc(SIZEFLOAT * dimension);

    for (short i = 0; i < dimension; ++i)
    {
        curandState state;
        curand_init(Randomize(randomSeed * (randomSeed + i), threadIdx.x), 0, 1, &state);
        position[i] = curand_uniform(&state) * 40.f - 20.f;
        velocity[i] = curand_uniform(&state) * 2.f - 1.f;
        if (position[i] < boundLeft)
        {
            position[i] = boundLeft;
        }
        else if (position[i] > boundRight)
        {
            position[i] = boundRight;
        }
    }

    float localBest = Fitness(position, dimension);
    float* positionBest = position;

    positions[localId] = position;
    velocities[localId] = velocity;
    fitnesses[localId] = localBest;

    float* row = (float*)((char*)bestPositions + localId * sizeof(float) * dimension);
    for (short j = 0; j < dimension; ++j)
    {
        row[j] = positionBest[j];
    }

    *bestGlobalValue = LONG_MAX;
}

__global__  void UpdateGlobalValues(const unsigned swarmSize, const short dimension, float* fitnesses,float** bestPositions, float* bestGlobalValue, float* bestGlobalPosition)
{
    int index = -1;
    for (int i = 0; i < swarmSize; ++i)
    {
        if (fitnesses[i] < *bestGlobalValue)
        {
            index = i;
        }
    }

    if (index != -1)
    {
        *bestGlobalValue = fitnesses[index];
        float* row = (float*)((char*)bestPositions + index * sizeof(float) * dimension);
        for (short j = 0; j < dimension; ++j)
        {
            bestGlobalPosition[j] = row[j];
        }
    }
}

__global__  void UpdatePositionAndVelocity(float** positions, float** velocities, float* fitnesses, float** bestPositions, float* bestGlobalPosition,const  float r1,const  float r2, const short dimension)
{
    int localId = threadIdx.x +  blockIdx.x * blockDim.x;
    float* velocityToUpdate = velocities[localId];
    float* row = (float*)((char*)bestPositions + localId * sizeof(float) * dimension);
    for (int i = 0; i < dimension; ++i)
    {
        velocityToUpdate[i] = W* velocityToUpdate[i] + C1 * r1 * (row[i] - positions[localId][i])
            + C2 * r2 * (*bestGlobalPosition - positions[localId][i]);
        positions[localId][i] += velocityToUpdate[i];
        if (positions[localId][i] < boundLeft)
        {
            positions[localId][i] = boundLeft;
        }
        else if (positions[localId][i] > boundRight)
        {
            positions[localId][i] = boundRight;
        }
    }
 
    float currentFitness = Fitness(positions[localId], dimension);
    if ( currentFitness < fitnesses[localId])
    {
        fitnesses[localId] = currentFitness;
 
        for (short j = 0; j < dimension; ++j)
        {
            row[j] = positions[localId][j];
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

    unsigned threadBlocks = swarmSize / 2;
    //Run kernel function
    Init<<<1, swarmSize >>> (positions, velocities, fitness, bestPositions,bestGlobalValue, r1, dimension);

    UpdateGlobalValues <<<1, 1 >>> (swarmSize, dimension, fitness, bestPositions, bestGlobalValue, bestGlobalPosition);

    for (int i = 0; i < iterations; ++i)
    {
        UpdatePositionAndVelocity <<<threadBlocks,2>>> (positions, velocities, fitness, bestPositions, bestGlobalPosition, r1, r2, dimension);
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
