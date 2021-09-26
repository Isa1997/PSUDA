import ctypes
import numpy as np
import time
from ctypes import *
from matplotlib import pyplot as plt
from random import random
import copy

def getFitnessFunction():
    dll = ctypes.windll.LoadLibrary("cudalib/SyncPSO.dll")
    dll.FitnessFunction.restype = ctypes.c_float
    func = dll.FitnessFunction
    func.argtypes = [POINTER(c_float), c_short]
    return func

# ---- Start Globals ----
SWARM_SIZE = 100
MAX_ITER = 10000
DIMENSION = 1

fitnessFunction = getFitnessFunction()
#---- End Globals ----

def getSyncPSO():
    dll = ctypes.windll.LoadLibrary("cudalib/SyncPSO.dll") 
    func = dll.SyncPSO
    func.argtypes = [c_uint, c_uint, c_float, c_float, POINTER(c_float), POINTER(c_float), c_short]
    return func

def SyncPSO():
    f = getSyncPSO()
    swarmSize = c_uint(SWARM_SIZE)
    numIterations = c_uint(MAX_ITER)
    max_out = c_float(10)
    r1 = c_float(random())
    r2 = c_float(random())

    solution = np.ones(DIMENSION).astype('float32')
    solution_ptr = solution.ctypes.data_as(POINTER(c_float))

    start = time.time()
    f(swarmSize, numIterations, r1, r2, pointer(max_out), solution_ptr, DIMENSION)
    end = time.time()

    print("***** Sync PSO *****")
    print("SyncPSO solution found: ", max_out.value)
    print('SyncPSO best position found {}'.format(solution))
    print("SyncPSO time: ", end-start)

    return end-start

def getRingPSO():
    dll = ctypes.windll.LoadLibrary("cudalib/RingPSO.dll")
    func = dll.RingPSO
    func.argtypes = [c_uint, c_uint, c_float, c_float, POINTER(c_float), POINTER(c_float), c_short]
    return func

def RingPSO():
    f = getRingPSO()
    swarmSize = c_uint(SWARM_SIZE)
    numIterations = c_uint(MAX_ITER)
    max_out = c_float(10)
    r1 = c_float(random())
    r2 = c_float(random())

    solution = np.ones(DIMENSION).astype('float32')
    solution_ptr = solution.ctypes.data_as(POINTER(c_float))

    start = time.time()
    f(numIterations, swarmSize, r1, r2, pointer(max_out), solution_ptr, c_short(DIMENSION))
    end = time.time()

    print("***** Ring PSO *****")
    print("RingPSO solution found: ", max_out.value)
    print('RingPSO best position found {}'.format(solution))
    print("RingPSO time: ", end-start)

    return end-start

class Particle:

    def __init__(self, w=0.7298, c1=1.49, c2=1.49, dim = DIMENSION):
        self.position = np.array([random() * 10.0 - 5.0 for i in range (0,dim)]).astype('float32')
        self.velocity = np.array([random() * 1.0 - 0.5 for i in range (0,dim)]).astype('float32')

        self.bestPosition = self.position
        self.currentValue = fitnessFunction(self.position.ctypes.data_as(POINTER(c_float)), c_short(len(self.position)))
        self.bestValue = self.currentValue
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def updatePosition(self, globalBestPosition, globalBestValue):

        self.position += self.velocity

        self.currentValue = fitnessFunction(self.position.ctypes.data_as(POINTER(c_float)), c_short(len(self.position)))
        if self.currentValue < self.bestValue:
            self.bestValue = self.currentValue
            self.bestPosition = self.position

            if self.currentValue < globalBestValue:
                globalBestValue = self.currentValue
                globalBestPosition = copy.copy(self.position)

        return globalBestPosition, globalBestValue

    def updateVelocity(self, globalBestPosition):
            r1 = random()
            r2 = random()
            cognitive_velocity = r1 * self.c1 * (self.bestPosition - self.position)
            social_velocity = r2 * self.c2 * (globalBestPosition - self.position)
            self.velocity = self.w * self.velocity + cognitive_velocity + social_velocity


def StandardPSO(dim = 1):
    start = time.time()
    swarm = [Particle(dim=dim) for _ in range(SWARM_SIZE)]
    globalBestPosition = copy.copy(swarm[0].position)
    globalBestValue = swarm[0].currentValue
    for particle in swarm:
        if particle.currentValue < globalBestValue:
            globalBestValue = particle.currentValue
            globalBestPosition = copy.copy(particle.position)
    bests = []
    for i in range(MAX_ITER):
        for j in range(len(swarm)):
            swarm[j].updateVelocity(globalBestPosition)
            swarm[j].updatePosition(globalBestPosition, globalBestValue)
        bests.append(globalBestValue)
    end = time.time()

    print("***** Standard PSO *****")
    print('StandardPSO solution found {}'.format( globalBestValue))
    print('StandardPSO best position found {}'.format( globalBestPosition))
    print("StandardPSO time: ", end-start)

    return end-start

if __name__ == "__main__":
    print("-------------------")
    print("Number of iterations: ", MAX_ITER)
    print("Number of particles in swarm: ", SWARM_SIZE)
    print("Problem Dimension: ", DIMENSION)
    print("f(x,y) = -std::fabs(std::sin(x)*std::cos(y)*std::exp(std::fabs(1-(std::sqrt(x*x + y*y)/3.1415))))")
    print("-------------------")
    DIMENSION = 2
    MAX_ITER = 10000
    SWARM_SIZE = 500
    SyncPSO()
    RingPSO()
    StandardPSO(dim=2)
    SyncPSO()
    RingPSO()
    StandardPSO(dim=2)

    '''
    r = range(1, 800, 100)
    timesSync = []
    timesRing = []
    timesStand = []
    MAX_ITER = 10000
    DIMENSION = 3
    for i in r:
        SWARM_SIZE = i
        print(i)
        timesSync.append(SyncPSO())
        timesRing.append(RingPSO())
        timesStand.append(StandardPSO())

    plt.plot(r, timesStand, color="red", label='Standard PSO')
    plt.plot(r, timesSync, color="blue", label='Sync PSO')
    plt.plot(r, timesRing, color="green", label='Ring PSO')
    plt.xlabel("Swar Size")
    plt.ylabel("Time in s")
    plt.legend()
    plt.show()'''

