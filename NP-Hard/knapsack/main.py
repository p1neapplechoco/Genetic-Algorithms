from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

DATA_FOLDER = "./data/"
MAX_POPULATION = 500
ELITE_INDIVIDUALS = 50
NUM_OF_GENERATIONS = 20
MUTATION_CHANCE = 0.2


def getProblemInfos(PROBLEM: int) -> Tuple[int, List[int], List[int], List[int]]:
    problem = f"p{PROBLEM:02d}"

    with open(DATA_FOLDER + problem + "_c.txt") as f:
        capacity = int(f.read())

    with open(DATA_FOLDER + problem + "_p.txt") as f:
        items = [int(p) for p in f.readlines()]

    with open(DATA_FOLDER + problem + "_w.txt") as f:
        weights = [int(w) for w in f.readlines()]

    with open(DATA_FOLDER + problem + "_s.txt") as f:
        solution = [int(s) for s in f.readlines()]

    return capacity, items, weights, solution


def relativityToSolution(
    ans: List[int], items: List[int], solution: List[int]
) -> float:
    ans_profit = 0
    sol_profit = 0

    for idx in range(len(items)):
        if ans[idx] == 1:
            ans_profit += items[idx]
        if solution[idx] == 1:
            sol_profit += items[idx]

    return (ans_profit / sol_profit) * 100


def initialize(num_of_things: int) -> List[List[int]]:
    Population = list()

    for inv in range(MAX_POPULATION):
        inv = list()
        for n in range(num_of_things):
            inv.append(np.random.randint(2))
        Population.append(inv)

    return Population


def calculate_fitness(
    individual: List[int], capacity: int, items: List[int], weights: List[int]
) -> int:
    cap, pf = 0, 0

    for idx, val in enumerate(individual):
        if val == 1:
            cap += weights[idx]
            pf += items[idx]

    if cap > capacity:
        return 0
    return pf


def mate(parent_1: List[int], parent_2: List[int]) -> List[int]:
    n = len(parent_1)

    child = list()

    for i in range(n):
        p = np.random.random(1)

        if p < (1 - MUTATION_CHANCE) / 2:
            child.append(parent_1[i])
        elif p < (1 - MUTATION_CHANCE):
            child.append(parent_2[i])
        else:
            child.append(np.random.randint(2))

    return child


def geneticAlgorithms(
    capacity: int, items: List[int], weights: List[int], solution: List[int]
):
    Population = initialize(len(items))

    for gen in range(NUM_OF_GENERATIONS):
        next_generation = list()
        mated = set()

        Population.sort(
            key=lambda x: calculate_fitness(x, capacity, items, weights), reverse=True
        )

        for elite in range(ELITE_INDIVIDUALS):
            next_generation.append(Population[elite])

        for p1 in range(len(Population)):
            if p1 in mated:
                continue
            for p2 in range(len(Population)):
                if p1 == p2 or p2 in mated:
                    continue
                num_of_kids = np.random.randint(1, 4)
                while num_of_kids and len(next_generation) <= MAX_POPULATION:
                    kid = mate(Population[p1], Population[p2])
                    next_generation.append(kid)
                    num_of_kids -= 1

        Population = next_generation.copy()

    Population = sorted(
        Population,
        key=lambda x: calculate_fitness(x, capacity, items, weights),
        reverse=True,
    )
    print(
        f"Evolution Answer: {Population[0]}, relativity to true answer: {relativityToSolution(Population[0], items, solution)}%"
    )


def bruteforce(
    capacity: int, items: List[int], weights: List[int], solution: List[int]
):
    def dfs(W, idx):
        if idx == len(items) or W == 0:
            return 0, [0] * len(items)

        if weights[idx] > W:
            return dfs(W, idx + 1)

        profit_pick, comb_pick = dfs(W - weights[idx], idx + 1)
        profit_pick += items[idx]
        comb_pick[idx] = 1

        profit_not_pick, comb_not_pick = dfs(W, idx + 1)

        if profit_pick > profit_not_pick:
            return profit_pick, comb_pick
        else:
            return profit_not_pick, comb_not_pick

    best_profit, best_combination = dfs(capacity, 0)

    print(
        f"Recursive Answer: {best_combination}, relativity to true answer: {relativityToSolution(best_combination, items, solution)}%"
    )


if __name__ == "__main__":
    # start = time.time()
    # PROBLEM = 11
    # capacity, items, weights, solution = getProblemInfos(PROBLEM)
    # print(f"<< PROBLEM {PROBLEM}: >>")
    # bruteforce(capacity, items, weights, solution)

    # end = time.time()
    # print(f"Elapsed time: {end - start}s")

    bf_times = defaultdict(float)
    ga_times = defaultdict(float)

    for i in range(1, 14):
        print(f"<< PROBLEM {i} >>")
        capacity, items, weights, solution = getProblemInfos(i)
        n = len(items)

        bf_start = time.time()
        bruteforce(capacity, items, weights, solution)
        bf_end = time.time()

        ga_start = time.time()
        geneticAlgorithms(capacity, items, weights, solution)
        ga_end = time.time()

        bf_times[n] = bf_end - bf_start
        ga_times[n] = ga_end - ga_start

    ns = sorted(bf_times.keys())
    bf_vals = [bf_times[n] for n in ns]
    ga_vals = [ga_times[n] for n in ns]

    plt.plot(ns, bf_vals, label="Brute-force", marker="o")
    plt.plot(ns, ga_vals, label="Genetic Algorithm", marker="o")

    plt.xlabel("Number of Items")
    plt.ylabel("Execution Time (s)")
    plt.title("Brute-force vs Genetic Algorithm Execution Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
