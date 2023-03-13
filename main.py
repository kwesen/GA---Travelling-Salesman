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
        nones = [i for i,_ in enumerate(O)]
        for i in nones:
            O[i] = P1[i]
    return O





# P1 = [3,4,5,6,7,8]
# P2 = [8,5,6,7,3,4]
# P1 = [2,4,5,1,3]
# P2 = [1,5,4,2,3]

# print(P2)
print(cycle_cross_over(P1,P2))