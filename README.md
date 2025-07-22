# Genetic Algorithms (GAs)

## Brief Introduction
- GAs draw inspiration from the process of natural selection—a mechanism Charles Darwin popularized after observing species variation during his 1835 voyage on the HMS Beagle. For example, the Galápagos penguin (Spheniscus mendiculus), the only penguin species living north of the equator, has evolved compact bodies and specialized salt‑removal glands to thrive in the tropical archipelago’s mix of semi‑arid and savanna climates. This example mirrors how GAs iteratively refine candidate solutions through selection, crossover, and mutation to optimize performance over successive generations.
- GAs simulate the process of natural selection which means those species that can adapt to changes in their environment can survive and reproduce and go to the next generation. In simple words, they simulate "survival of the fittest" among individuals of generations to solve a problem.

## Main Motivation
1. Individuals of a generation compete for a selected task.
2. Successful individuals then mate to produce offspring.
3. Genetic traits are passed on, and weaker traits are gradually replaced by stronger ones in later generations.
4. Over successive generations, the population’s performance on the task improves.

### Fitness score
- Fitness score is used to calculate the ability of a individual.
- The higher their fitness score, the higher their chance of selection for reproduction
- Later generations *typically* improve the fitness score.

### Operators
- Selection: give preference to individuals with good fitness scores and allow them to pass their genes onto the successive generation.
- Crossover: representing the process of mating between two selcted individuals through the selection operator. 
- Mutation: main idea is to insert random genes in offspring to maintain the diversity of the population.

### Summarized process
1. Initialize populations
2. Calculate the fitness of the population.
3. "Evolution process":
	a. Select successful parents.
	b. Mating process.
	c. Mutation process.
	d. Goto 2 until convergence (when the task is solved or after a number of generations).

## Implementation

### Introduction Task (Guessing Sentence)