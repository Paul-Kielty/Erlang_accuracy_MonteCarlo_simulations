from math import factorial

def offeredTraffic(calls_per_hour, hours_per_call): # A0
    return calls_per_hour*hours_per_call
    
def erlangb(n, A0):
    denom = 0
    for i in range(n):
        denom += (A0**i)/(factorial(i))
    E1 = (A0**n)/(factorial(n))/denom
    return E1

print(erlangb(39,30))
print(erlangb(4, 0.595))
