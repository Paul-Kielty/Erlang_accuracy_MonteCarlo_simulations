from math import factorial
import matplotlib.pyplot as plt
import numpy as np
MEAN_CALL_HOLDING_TIME = 2/60 # HRS

def offeredTraffic(calls_per_hour, hours_per_call): # A0
    return calls_per_hour*hours_per_call

def erlangb(n, A0):
    denom = 0
    for i in range(n):
        denom += (A0**i)/(factorial(i))
    E1 = (A0**n)/(factorial(n))/denom
    return E1

# print(erlangb(39,30))
# print(erlangb(4, 0.595))

def trafficSimulation(numCalls):
    calls = np.random.exponential(scale=MEAN_CALL_HOLDING_TIME, size=numCalls)
    print(calls.mean())
    plt.hist(calls, bins=1000)
    # plt.plot(calls)
    plt.title("Call duration frequency")
    # plt.ylabel=("frequency")
    plt.show()

trafficSimulation(10000)