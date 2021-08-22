
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand.h"
#include "curand_kernel.h"
#include <limits>
#include "Test.h"


#define DELLEXPORT extern "C" __declspec(dllexport)

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
    return static_cast<int>(seed*100) + threadId;
}

__global__  void kernel(unsigned iterations, unsigned swarmSize, float* bestPositions, float* s_fitnesses, float r1, float r2, float* max_out)
{
    //Shared memory
    __shared__ float s_maxGlobalFitness;
    __shared__ float s_bestGlobalPosition;
     
    if (threadIdx.x == 0) 
    {
        s_maxGlobalFitness = LONG_MAX;
    }

    //Init
    curandState state;
    curand_init(Randomize(r1, threadIdx.x), 0, 1, &state);
    float position = curand_uniform(&state) * 40.f - 20.f;
    float velocity = curand_uniform(&state) * 2.f - 1.f;
   
    float currentValue = Fitness(position);
    float localBest = currentValue;
    float positionBest = position;
    s_fitnesses[threadIdx.x] = localBest;
    bestPositions[threadIdx.x] = positionBest;
    __syncthreads();
    
    //Updating best globals before first iteration
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < swarmSize; ++i)
        {
            if (s_fitnesses[i] < s_maxGlobalFitness)
            {
                s_maxGlobalFitness = s_fitnesses[i];
                s_bestGlobalPosition = bestPositions[i];
            }
        }
    }
    __syncthreads();
    //Iterations
    for (int i = 0; i < iterations; ++i)
    {
        //update of velocity and positions
        velocity = W * velocity + C1 *r1* (positionBest - position) + C2 * r2*(s_bestGlobalPosition - position);
        position = position + velocity;

        //uptate swarm best position based on fitness
        if (Fitness(position) < localBest)
        {
            positionBest = position;
            localBest = Fitness(position);
            bestPositions[threadIdx.x] = positionBest;
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
                    s_bestGlobalPosition = bestPositions[i];
                }
            }
        }
        __syncthreads();
    }

    //updating out max
    if (threadIdx.x == 0)
    {
        *max_out = s_maxGlobalFitness;
    }
}

DELLEXPORT void SyncPSO(unsigned* swarmSize, unsigned* numIteration, float* R1, float* R2, float* max)
{
    //Init for GPU vars
    float* max_out;

    //Arrars
    float* fitness;
    float* bestPositions;

    cudaMalloc((void**)&max_out, SIZEFLOAT);
    cudaMalloc((void**)&bestPositions, SIZEFLOAT * (*swarmSize));
    cudaMalloc((void**)&fitness, SIZEFLOAT * (*swarmSize));

    //Run kernel function 
    kernel <<<1, *swarmSize>>> (*numIteration, *swarmSize, bestPositions, fitness, *R1, *R2, max_out);

    //Copy into main memory from GPU
    cudaMemcpy(max, max_out, SIZEFLOAT, cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(bestPositions); 
    cudaFree(max_out);
    cudaFree(fitness);
}
