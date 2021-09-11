
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand.h"
#include "curand_kernel.h"
#include <limits>
#include "Test.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>

#define DELLEXPORT extern "C" __declspec(dllexport)

#define W (0.75f)
#define C1 (1.f)
#define C2 (2.f)

#define SIZEFLOAT (sizeof(float))
#define SIZEINT (sizeof(int))
#define SIZEUINT (sizeof(unsigned))

__device__
float Fitness(float* x, const short dimension)
{
    return FitnessFunction(x, dimension);
}

__device__
int Randomize(float seed, int threadId) 
{
    return static_cast<int>(seed*100) + threadId;
}

__global__  void kernel(unsigned iterations, unsigned swarmSize, float** bestPositions, float* s_fitnesses, float r1, float r2, float* max_out, float* bestPosition, const short dimension)
{
    //Shared memory
    __shared__ float s_maxGlobalFitness;
    __shared__ float* s_bestGlobalPosition;
     
    if (threadIdx.x == 0) 
    {
        s_maxGlobalFitness = LONG_MAX;
        s_bestGlobalPosition = (float*)malloc(sizeof(float) * dimension);
    }

    //Init
    float* position = (float*)malloc(SIZEFLOAT * dimension);
    for (short i = 0; i < dimension; ++i)
    {
        curandState state;
        curand_init(Randomize(r1, threadIdx.x), 0, 1, &state);
        position[i] = curand_uniform(&state) * 40.f - 20.f;
    }

    float* velocity = (float*)malloc(SIZEFLOAT * dimension);
    for (short i = 0; i < dimension; ++i)
    {
        curandState state;
        curand_init(Randomize(r1, threadIdx.x), 0, 1, &state);
        velocity[i] = curand_uniform(&state) * 2.f - 1.f;
    }

    float currentValue = Fitness(position,dimension);
    float localBest = currentValue;
    float* positionBest = position;
    s_fitnesses[threadIdx.x] = localBest;

    float* row = (float*)((char*)bestPositions + threadIdx.x * sizeof(float) * dimension);
    for (short i = 0; i < dimension; ++i)
    {
        row[i] = position[i];
    }
    __syncthreads();
    
    //Updating best globals before first iteration
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < swarmSize; ++i)
        {
            if (s_fitnesses[i] < s_maxGlobalFitness)
            {
                s_maxGlobalFitness = s_fitnesses[i];

                float* row = (float*)((char*)bestPositions + i * sizeof(float) * dimension);
                for (short j = 0; j < dimension; ++j)
                {
                    s_bestGlobalPosition[j] = row[j];
                }
            }
        }
    }
    __syncthreads();
    //Iterations
    for (int i = 0; i < iterations; ++i)
    {
        //update of velocity and positions
        for (short j = 0; j < dimension; ++j)
        {
            velocity[j] = W * velocity[j] + C1 * r1 * (positionBest[j] - position[j]) + C2 * r2 * (s_bestGlobalPosition[j] - position[j]);
            position[j] = position[j] + velocity[j];
        }

        //uptate swarm best position based on fitness
        float currentFitness = Fitness(position, dimension);
        if ( currentFitness < localBest)
        {
            positionBest = position;
            localBest = currentFitness;
            float* row = (float*)((char*)bestPositions + threadIdx.x * sizeof(float) * dimension);
            for (short i = 0; i < dimension; ++i)
            {
                row[i] = position[i];
            }
            s_fitnesses[threadIdx.x] = localBest;
        }
        __syncthreads();

        //update global max by first thread
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < swarmSize; i++)
            {
                if (s_fitnesses[i] < s_maxGlobalFitness)
                {
                    s_maxGlobalFitness = s_fitnesses[i];
                    float* row = (float*)((char*)bestPositions + i * sizeof(float) * dimension);
                    for (short j = 0; j < dimension; ++j)
                    {
                        s_bestGlobalPosition[j] = row[j];
                    }
                }
            }
        }
        __syncthreads();
    }

    //updating out max
    if (threadIdx.x == 0)
    {
        *max_out = s_maxGlobalFitness;
        
        for (int i = 0; i < dimension; ++i)
        {
            bestPosition[i] = s_bestGlobalPosition[i];
        }
    }
}

DELLEXPORT void SyncPSO(unsigned swarmSize, unsigned numIteration, float R1, float R2, float* max, float* bestPosition, const short dimension )
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
    float* max_out;
    float* bestPosition_out;

    //Arrars
    float* fitness;
    float** bestPositions;

    size_t pitch;
    cudaMallocPitch((void**)&bestPositions, &pitch, SIZEFLOAT * dimension, swarmSize);
    cudaMalloc((void**)&max_out, SIZEFLOAT);
    cudaMalloc((void**)&bestPosition_out, SIZEFLOAT * dimension);
    cudaMalloc((void**)&fitness, SIZEFLOAT * swarmSize);

    //Run kernel function 
    kernel <<<1, swarmSize>>> (numIteration, swarmSize, bestPositions, fitness, R1, R2, max_out, bestPosition_out, dimension);
    
    //Copy into main memory from GPU
    cudaMemcpy(max, max_out, SIZEFLOAT, cudaMemcpyDeviceToHost);
    cudaMemcpy(bestPosition, bestPosition_out, SIZEFLOAT * dimension, cudaMemcpyDeviceToHost);
    
    //Free GPU memory
    cudaFree(bestPositions); 
    cudaFree(max_out);
    cudaFree(bestPosition_out);
    cudaFree(fitness);
}
