import ctypes
from ctypes import * 
import numpy as np
from random import random
import time

def getSyncPSO():
    dll = ctypes.windll.LoadLibrary("cudalib/SyncPSO.dll") 
    func = dll.SyncPSO
    func.argtypes = [POINTER(c_uint),POINTER(c_uint),POINTER(c_float), POINTER(c_float), POINTER(c_float)] 
    return func

def SyncPSO():
    f = getSyncPSO()
    swarmSize = c_uint(300)
    numIterations = c_uint(1000)
    max_out = c_float(10)
    r1 = c_float(random())
    r2 = c_float(random())

    it = np.zeros(100).astype('float32')
    it_p = it.ctypes.data_as(POINTER(c_float))

    f(pointer(swarmSize), pointer(numIterations),pointer(r1),pointer(r2), pointer(max_out))

    print("Number of iterations: ", numIterations.value)
    print("Number of particles in swarm: ", swarmSize.value)
    
    print("Solution Found: ", max_out.value)


def getRingPSO():
    dll = ctypes.windll.LoadLibrary("cudalib/RingPSO.dll")
    func = dll.RingPSO
    func.argtypes = [c_uint,c_uint,c_float, c_float, POINTER(c_float)]
    return func

def RingPSO():
    f = getRingPSO()
    swarmSize = c_uint(300)
    numIterations = c_uint(1000)
    max_out = c_float(10)
    r1 = c_float(random())
    r2 = c_float(random())

    it = np.zeros(100).astype('float32')
    it_p = it.ctypes.data_as(POINTER(c_float))

    f(swarmSize, numIterations, r1, r2, pointer(max_out))

    print("Number of iterations: ", numIterations.value)
    print("Number of particles in swarm: ", swarmSize.value)
    print("Solution Found: ", max_out.value)

if __name__ == "__main__":
    start = time.time()
    SyncPSO();
    end = time.time()
    print("SyncPSO time: ", end-start)
    start = time.time()
    RingPSO();
    end = time.time()
    print("RingPSO time: ", end-start)
