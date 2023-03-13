import numpy as np

def cycle_cross_over(P1: list,P2: list) -> list[int]:
    O: list = [None for i in range(len(P1))]
    O[0] = P1[0]

    val = P2[P1.index(O[0])]
    index = P1.index(val)
    val1 = P2[index]
    O[index] = val 

    while val1 not in O:
        O[index] = val   
        val1 = P2[index]
        val = P1[P2.index(val1)]
        index = P1.index(val1)

    if None in O:
        nones = [i for i,val in enumerate(O) if val is None]
        for i in nones:
            O[i] = P1[i]
    return O

def mutation(P: list) -> list[int]:
    chosen = np.random.choice(P,2, replace=False)
    i1 = P.index(chosen[0])
    i2 = P.index(chosen[1])
    P[i1],P[i2] = P[i2],P[i1]
    return P




# P1 = [3,4,5,6,7,8]
# P2 = [8,5,6,7,3,4]
# P1 = [2,4,5,1,3]
# P2 = [1,5,4,2,3]
P1 = [1,2,3,4,5]
P2 = [5,2,3,1,4]
# print(P2)
cross = cycle_cross_over(P1,P2)
print(cross)
# test = mutation(P1.copy())