#pragma once

#include <string>

int random_int(int low, int high);       // For initialize an individual or mutated genes creation.
double random_double(int low, int high); // For each gene's probability to be carried through mating.

class Individual
{
public:
    std::string genes_ = "";
    int loss_score_ = 0;

public:
    Individual() = default;

    Individual(const int &);
    Individual(const std::string &);
    Individual(const Individual &);

    bool operator<(const Individual &) const;
    bool operator>(const Individual &) const;

    Individual &operator=(const Individual &);

    Individual mate(const Individual &);
    void calculate_loss(const std::string &);
};