#pragma once
#include <random>

namespace RandomGenerators {
template <typename rand_T>
class RealGenerator {
  public:
    // Constructors
    RealGenerator(rand_T min, rand_T max) : m_dist(min, max) {
        std::random_device rd;
        m_seed = rd();
        m_rng.seed(m_seed);
    }

    RealGenerator(rand_T min, rand_T max, unsigned int seed)
        : m_dist(min, max), m_seed(seed) {
        m_rng.seed(m_seed);
    }

    // Generator fun
    rand_T generate_random() { return m_dist(m_rng); }

  private:
    // Type of random number distribution
    std::uniform_real_distribution<rand_T> m_dist;

    // Mersenne Twister: Good quality random number generator
    std::mt19937 m_rng;

    // Seed
    unsigned int m_seed;
};

template <typename rand_T>
class IntGenerator {
  public:
    // Constructors
    IntGenerator(rand_T min, rand_T max) : m_dist(min, max) {
        std::random_device rd;
        m_seed = rd();
        m_rng.seed(m_seed);
    }

    IntGenerator(rand_T min, rand_T max, unsigned int seed)
        : m_dist(min, max), m_seed(seed) {
        m_rng.seed(m_seed);
    }

    // Generator fun
    rand_T generate_random() { return m_dist(m_rng); }

  private:
    // Type of random number distribution
    std::uniform_int_distribution<rand_T> m_dist;

    // Mersenne Twister: Good quality random number generator
    std::mt19937 m_rng;

    // Seed
    unsigned int m_seed;
};
}  // namespace RandomGenerators