#include "individual.h"
#include <string>
#include <random>

static const std::string ALPHABET =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    " .,!?;:'\"-";

std::random_device rd;
std::mt19937 gen(rd());

// Helper functions
int random_int(int low, int high) // For initialize an individual or mutated genes creation.
{
    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

char random_letter()
{
    int max_idx = static_cast<int>(ALPHABET.length()) - 1;
    return ALPHABET[random_int(0, max_idx)];
}

double random_double(int low, int high) // For each gene's probability to be carried through mating.
{
    std::uniform_real_distribution<> dist(low, high);
    return dist(gen);
}

// Individual Class
Individual::Individual(const int &length_of_genes) : Individual()
{
    for (int i = 0; i < length_of_genes; i++)
        genes_ += static_cast<char>(random_int(CHAR_MIN, CHAR_MAX));
}

Individual::Individual(const std::string &genes)
{
    genes_ = genes;
}

Individual::Individual(const Individual &o)
{
    this->genes_ = o.genes_;
    this->loss_score_ = o.loss_score_;
}

bool Individual::operator<(const Individual &o) const
{
    return this->loss_score_ < o.loss_score_;
}

bool Individual::operator>(const Individual &o) const
{
    return this->loss_score_ > o.loss_score_;
}

Individual &Individual::operator=(const Individual &o)
{
    if (this == &o)
        return *this;

    this->genes_ = o.genes_;
    this->loss_score_ = o.loss_score_;
    return *this;
}

Individual Individual::mate(const Individual &o)
{
    std::string successor_genes = "";

    for (int i = 0; i < genes_.length(); i++)
    {
        double prob = random_double(0, 1);

        if (prob < 0.45)
            successor_genes += this->genes_[i];

        else if (prob < 0.9)
            successor_genes += o.genes_[i];

        else
            successor_genes += static_cast<char>(random_int(CHAR_MIN, CHAR_MAX));
    }

    return Individual(successor_genes);
}

void Individual::calculate_loss(const std::string &task)
{
    loss_score_ = 0;
    for (size_t i = 0; i < genes_.size(); i++)
    {
        if (genes_[i] != task[i])
            ++loss_score_;
    }
}