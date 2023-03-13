import numpy as np

def cycle_cross_over(P1: list,P2: list) -> list[int]:
    Offspring: list = [None for i in range(len(P1))]
    index = 0
    val_t_find = ''
    while val_t_find not in Offspring:
        val_insert = P1[index]
        Offspring[index] = val_insert
        val_t_find = P2[index]
        index = P1.index(val_t_find)

    # replace None with P2 values    
    if None in Offspring:
        nones = [i for i,val in enumerate(Offspring) if val is None]
        for i in nones:
            Offspring[i] = P2[i]
    return Offspring

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
expected = [2,5,4,1,3]
# P1 = [3,4,1,2,5]
# P2 = [4,2,5,1,3]
# print(P2)
cross = cycle_cross_over(P1,P2)
print(cross)
# test = mutation(P1.copy())