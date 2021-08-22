import ctypes
import numpy as np
import time
from ctypes import *
from matplotlib import pyplot as plt
from random import random

def getFitnessFunction():
    dll = ctypes.windll.LoadLibrary("cudalib/SyncPSO.dll")
    dll.FitnessFunction.restype = ctypes.c_float
    func = dll.FitnessFunction
    func.argtypes = [c_float]
    return func

# ---- Start Globals ----
SWARM_SIZE = 100
MAX_ITER = 10000
fitnessFunction = getFitnessFunction()
#---- End Globals ----

def getSyncPSO():
    dll = ctypes.windll.LoadLibrary("cudalib/SyncPSO.dll") 
    func = dll.SyncPSO
    func.argtypes = [POINTER(c_uint), POINTER(c_uint), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
    return func

def SyncPSO():
    f = getSyncPSO()
    swarmSize = c_uint(SWARM_SIZE)
    numIterations = c_uint(MAX_ITER)
    max_out = c_float(10)
    r1 = c_float(random())
    r2 = c_float(random())

    #it = np.zeros(100).astype('float32')
    #it_p = it.ctypes.data_as(POINTER(c_float))

    start = time.time()
    f(pointer(swarmSize), pointer(numIterations),pointer(r1),pointer(r2), pointer(max_out))
    end = time.time()

    print("Number of iterations: ", numIterations.value)
    print("Number of particles in swarm: ", swarmSize.value)
    print("SyncPSO solution found: ", max_out.value)
    print("SyncPSO time: ", end-start)
    return end-start

def getRingPSO():
    dll = ctypes.windll.LoadLibrary("cudalib/RingPSO.dll")
    func = dll.RingPSO
    func.argtypes = [c_uint,c_uint,c_float, c_float, POINTER(c_float)]
    return func

def RingPSO():
    f = getRingPSO()
    swarmSize = c_uint(SWARM_SIZE)
    numIterations = c_uint(MAX_ITER)
    max_out = c_float(10)
    r1 = c_float(random())
    r2 = c_float(random())

    it = np.zeros(100).astype('float32')

    start = time.time()
    f(numIterations, swarmSize, r1, r2, pointer(max_out))
    end = time.time()

    print("Number of iterations: ", numIterations.value)
    print("Number of particles in swarm: ", swarmSize.value)
    print("RingPSO solution found: ", max_out.value)
    print("RingPSO time: ", end-start)
    return end-start

class Particle:

    def __init__(self, w=0.75, c1=1, c2=2):
        self.position = random() * 40.0 - 20.0
        self.velocity = random() * 2.0 - 1.0

        self.bestPosition = self.position
        self.currentValue = fitnessFunction(self.position)
        self.bestValue = self.currentValue

        self.w = w
        self.c1 = c2
        self.c2 = c2

    def updatePosition(self, globalBestPosition, globalBestValue):

        self.position += self.velocity

        self.currentValue = fitnessFunction(self.position)
        if self.currentValue < self.bestValue:
            self.bestValue = self.currentValue
            self.bestPosition = self.position

            if self.currentValue < globalBestValue:
                globalBestValue = self.currentValue
                globalBestPosition = self.position

        return globalBestPosition, globalBestValue

    def updateVelocity(self, globalBestPosition):
            r1 = random()
            r2 = random()
            cognitive_velocity = r1 * self.c1 * (self.bestPosition - self.position)
            social_velocity = r2 * self.c2 * (globalBestPosition - self.position)
            self.velocity = self.w * self.velocity + cognitive_velocity + social_velocity


def StandardPSO():
    start = time.time()
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    globalBestPosition = swarm[0].position
    globalBestValue = swarm[0].currentValue
    for particle in swarm:
        if particle.currentValue < globalBestValue:
            globalBestValue = particle.currentValue
            globalBestPosition = particle.position

    bests = []
    for i in range(MAX_ITER):
        for j in range(len(swarm)):
            swarm[j].updateVelocity(globalBestPosition)
            globalBestPosition, globalBestValue = swarm[j].updatePosition(globalBestPosition, globalBestValue)

        bests.append(globalBestValue)
    end = time.time()
    print('StandardPSO solution: {}, value: {}'.format(globalBestPosition, globalBestValue))
    print("StandardPSO time: ", end-start)
    return end-start

if __name__ == "__main__":
    SyncPSO()
    RingPSO()
    StandardPSO()

    #timesSync = []
    #timesRing = []
   # timesStand = []
    #MAX_ITER = 100
    #for i in range(1, 1000,1):
     #   SWARM_SIZE = i
      #  timesSync.append(SyncPSO())
       # timesRing.append(RingPSO())
        #timesStand.append(StandardPSO())

    #plt.plot(range(1,100,1), timesStand, color="red")
    #plt.plot(range(1,100,1), timesSync, color="blue")
    #plt.plot(range(1,100,1), timesRing, color="green")
    #plt.show()