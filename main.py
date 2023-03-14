import numpy as np
import matplotlib.pyplot as plt

def cycle_cross_over(P1: list,P2: list) -> list[int]:
    def get_child(P1: list,P2: list): 
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
    return get_child(P1,P2),get_child(P2,P1)

def mutation(P: list) -> list[int]:
    chosen = np.random.choice(P,2, replace=False)
    i1 = P.index(chosen[0])
    i2 = P.index(chosen[1])
    P[i1],P[i2] = P[i2],P[i1]
    return P

# x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
# y = [1, 4, 5, 3, 0 ,4, 10, 6, 9, 10]
# plt.plot(x,y,'ro')
# plt.show()

# P1 = [3,4,5,6,7,8]
# P2 = [8,5,6,7,3,4]
# P1 = [2,4,5,1,3]
# P2 = [1,5,4,2,3]
# expected = [2,5,4,1,3]
# P1 = [3,4,2,1,5]
# P2 = [4,1,5,2,3]
# print(P2)
# cross = cycle_cross_over(P1,P2)
# print(cross)
# test = mutation(P1.copy())


def city_cord_reader(path: str) -> list[tuple[int, int]]:
    '''Reads the file provided as a path that contains the coordinates of cities\n
    The coordinates should be written in seperate lines in [ ] square
    brackets and separated by spaces'''

    with open(path, 'r+') as file:
        usefull_data = file.readlines()[1:]
        ix1,ix2 = usefull_data[0].find('[')+1, usefull_data[0].find(']')
        iy1, iy2 = usefull_data[1].find('[')+1, usefull_data[1].find(']')
        x, y = usefull_data[0][ix1:ix2], usefull_data[1][iy1:iy2]
        x, y = x.split(' '), y.split(' ')
        cords = []
        for xval,yval in zip(x,y):
            cords.append((int(xval), int(yval)))
    return cords

cords : list[tuple] = city_cord_reader('Traveling Salesman Problem Data-20230314\cities_1.txt')

# for founder in cords:
#     x,y = founder
#     plt.plot(x,y, 'ro')

# plt.show()

def init_population(n: int, pop: int) -> list[list[int]]:
    grand_parent = [n for n in range(1,n+1)]
    parent = []
    for pop in range(pop):
        parent.append(np.random.permutation(grand_parent))
    return parent 

def euclidean_dist(city1: tuple, city2: tuple) -> float:
    x1, y1 = city1
    x2, y2 = city2
    dist = np.sqrt(np.power(x1-x2, 2) + np.power(y1-y2, 2))
    return dist

def city_dist_map(cities: list[tuple]) -> dict:
    distance_map = {i : {} for i in range(len(cities))}
    for key, city in enumerate(cities):
        for destkey, destination in enumerate(cities):
            distance_map[key][destkey] = euclidean_dist(city,destination)
    return distance_map

distances = city_dist_map(cords)
print(distances[1][2])
