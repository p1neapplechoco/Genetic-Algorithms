#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <random>
#include "individual.h"

#define MAX_POPULATION 1000
#define ELITE_INDIVIDUALS 999

const std::string task = "This is a secret message.";
const int length_of_genes = task.length();

void natural_selection(std::vector<Individual> &population)
{
    for (Individual &inv : population)
        inv.calculate_loss(task);
}

int main()
{
    std::vector<Individual> population;

    for (int i = 0; i < MAX_POPULATION; i++)
        population.push_back(Individual(length_of_genes));

    natural_selection(population);

    int num_of_generations = 0;

    while (population[0].genes_ != task)
    {
        std::unordered_set<int> mated;
        std::vector<Individual> successors;

        std::sort(population.begin(), population.end());

        std::cout << "<< Generation: " << num_of_generations++ << " >>" << std::endl;
        std::cout << "Best genes: " << population[0].genes_ << " , loss: " << population[0].loss_score_ << std::endl;

        for (int i = 0; i < ELITE_INDIVIDUALS; i++)
        {
            successors.push_back(population[i]);
        }

        for (int p1 = 0; p1 < (int)population.size() && successors.size() < MAX_POPULATION; ++p1)
        {
            if (mated.count(p1))
                continue;
            for (int p2 = p1 + 1; p2 < (int)population.size() && successors.size() < MAX_POPULATION; ++p2)
            {
                if (mated.count(p2))
                    continue;

                int num_of_children = random_int(1, 3);
                while (num_of_children-- && successors.size() < MAX_POPULATION)
                {
                    Individual child = population[p1].mate(population[p2]);
                    successors.push_back(child);
                }

                mated.insert(p1);
                mated.insert(p2);
                break;
            }

            if (successors.size() >= MAX_POPULATION)
                break;
        }

        population.clear();
        population = successors;

        natural_selection(population);
    }

    return 0;
}