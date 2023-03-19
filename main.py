import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import warnings

def timer(function):
    '''General purpose timing decorator\n
    Before using make sure perf_counter is imported'''
    def wrapper(*args,**kwargs):
        start = perf_counter()
        function(*args,**kwargs)
        stop = perf_counter()
        print(f'Execution took {stop-start}')
    return wrapper

def cycle_cross_over(P1: list,P2: list) -> list[list[int],list[int]]:
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
    return [get_child(P1,P2),get_child(P2,P1)]

def offspring(parents: list[list]) -> list[list[int]]:
    mothers = parents[:int(len(parents)/2)]
    fathers = parents[int(len(parents)/2):]
    kids = []
    for mother, father in zip(mothers, fathers):
        kids.extend(cycle_cross_over(mother, father))
    return kids

def mutation(P: list) -> list[int]:
    chosen = np.random.choice(P,2, replace=False)
    i1 = P.index(chosen[0])
    i2 = P.index(chosen[1])
    P[i1],P[i2] = P[i2],P[i1]
    return P

def random_mutation(population: list[list], mutation_chance: float) -> list[list]:
    for i in range(len(population)):
        will_it_mut = np.random.uniform(0,1)
        if will_it_mut <= mutation_chance:
            population[i] = mutation(population[i])
    return population

def city_cord_reader(path: str) -> list[tuple[float,float]]:
    '''Reads the file provided as a path that contains the coordinates of cities\n
    The coordinates should be written in seperate lines in [ ] square
    brackets and separated by spaces'''

    with open(path, 'r+') as file:
        usefull_data = file.readlines()[1:]
        ix1,ix2 = usefull_data[0].find('[')+1, usefull_data[0].find(']')
        iy1, iy2 = usefull_data[1].find('[')+1, usefull_data[1].find(']')
        x, y = usefull_data[0][ix1:ix2], usefull_data[1][iy1:iy2]
        x, y = x.split(' '), y.split(' ')
        x = [i for i in x if i]
        y = [i for i in y if i]
        if len(x)!=len(y):
            raise Exception("Lengths don't match, file may be corrupted")
        cords = []
        for xval,yval in zip(x,y):
            cords.append((float(xval), float(yval)))
    return cords

def init_population(n: int, pop: int) -> list[list[int]]:
    grand_parent = [n for n in range(1,n+1)]
    parent = []
    for pop in range(pop):
        parent.append(np.random.permutation(grand_parent).tolist())
    return parent 

def euclidean_dist(city1: tuple, city2: tuple) -> float:
    x1, y1 = city1
    x2, y2 = city2
    dist = np.sqrt(np.power(x1-x2, 2) + np.power(y1-y2, 2))
    return dist

def city_dist_map(cities: list[tuple]) -> dict:
    distance_map = {i+1 : {} for i in range(len(cities))}
    for key, city in enumerate(cities):
        for destkey, destination in enumerate(cities):
            distance_map[key+1][destkey+1] = euclidean_dist(city,destination)
    return distance_map

def cost_value_individual(distance_map: dict, chromosome: list[int]) -> float:
    cost = distance_map[chromosome[-1]][chromosome[0]]
    for i in range(len(chromosome)-1):
        cost += distance_map[chromosome[i]][chromosome[i+1]]
    return cost

def invert_cost(cost: float) -> float:
    cost_val = 1/cost
    return cost_val

def probability(tis:list[float]) -> list[float]:
    sum_tis = sum(tis)
    pie = [ti/sum_tis for ti in tis]
    return pie

def fitness(population: list[list],distance_map:dict):
    fis = [invert_cost(cost_value_individual(distance_map,pop)) for pop in population]
    # mf = max(fis)
    # tis = [mf - fi for fi in fis]
    tis = fis
    return tis

def proportional_selection(population:list[list], distance_map: dict, n:float):
    tis = fitness(population, distance_map)
    probs = probability(tis)
    parents = []
    for i in range(int(n*len(population))):
        pisum = 0
        j = 0
        r = np.random.uniform(0,1)
        while pisum <= r:
            pisum += probs[j]
            j += 1
        parents.append(population[j-1])
    return parents

def selection_of_best_individuals(population: list[list], distance_map: dict, population_count: int) -> list[list]:
    tis = fitness(population, distance_map)
    probs = probability(tis)
    best_individuals = [x for _,x in sorted(zip(probs,population), key= lambda pair: pair[0],reverse=True)] # nice general sorting thanks stack overflow <3        
    return best_individuals[:population_count]

def plot_path(city_coordinates: list[tuple], chromosome: list[int]) -> None:
    '''Takes the city coordinated as a list of (x,y) tuples
      as well as the chromosome that determines the path'''
    for i,city in enumerate(city_coordinates):
        x,y = city
        plt.plot(x,y,'ro')
        plt.text(x+0.1,y+0.1,str(i+1),color='#FF0000')

    coords = {i+1:city for i,city in enumerate(city_coordinates)}

    for i in range(len(chromosome[:-1])):
        x,y = coords[chromosome[i]] # start loc coords
        xi,yi = coords[chromosome[i+1]]
        plt.plot([x,xi],[y,yi],'k')

    x,y = coords[chromosome[0]]
    xi,yi = coords[chromosome[-1]]

    plt.plot([x,xi],[y,yi],'k')
    plt.axis('off')
    plt.show()

def test_cycle_cross_over(repeats):
    for i in range(repeats):
        tester = []
        population = init_population(10,300)
        pool1, pool2 = population[:150],population[150:]
        for specimen1,specimen2 in zip(pool1,pool2):
            uno,duo = cycle_cross_over(specimen1,specimen1)
            if len(set(uno))!=len(uno)  or len(set(duo))!=len(duo):
                raise Exception(f'Lengths dont match 1:{specimen1} 2:{specimen2} c1:{uno} c2:{duo}')
            if None in uno or None in duo:
                raise Exception(f'None detected 1:{specimen1} 2:{specimen2} c1:{uno} c2:{duo}')
    else:
        print('Success!!!')

def GA(path: str, population_count: int, mutation_chance: float, n: float, iterations: int):
    cities = city_cord_reader(path)
    distances = city_dist_map(cities)
    N = len(cities)
    Population = init_population(N,population_count)
    for iteration in range(iterations):
        Parents = proportional_selection(population = Population,
                                         distance_map = distances,
                                         n = n)
        Offspring = offspring(Parents)
        Offspring = random_mutation(Offspring, mutation_chance)
        Parents.extend(Offspring)
        Population = selection_of_best_individuals(Parents,distances,population_count)
    best = Population[0]
    # best = selection_of_best_individuals(Population,distances,population_count)[0]
    # best = Population[0]
    print(best)
    plot_path(cities,best)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    GA(path='Traveling Salesman Problem Data-20230314\cities_4.txt',
       population_count = 250,
       mutation_chance = 0.2,
       iterations = 1000,
       n = 0.8)
# test_cycle_cross_over(10000)

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

# cords : list[tuple] = city_cord_reader('Traveling Salesman Problem Data-20230314\cities_1.txt')
# distances = city_dist_map(cords)
# ch = [9, 1, 8, 10, 3, 2, 7, 6, 5, 4]
# print(cost_value_individual(distances,ch))
# print(distances[1][9])

# plot_path(cords,[1, 2, 3, 4, 9, 6, 7, 8, 5, 10])
# pop = init_population(6,100)
# herro = proportional_selection(pop,distances,0.8)
# print(len(herro))
# print(max(selection(pop,distances)))