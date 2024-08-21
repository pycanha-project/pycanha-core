#pragma once
#include <random>

namespace random_generators {
template <typename RandT>
class RealGenerator {
  public:
    // Constructors
    RealGenerator(RandT min, RandT max) : _dist(min, max) {
        std::random_device rd;
        _seed = rd();
        _rng.seed(_seed);
    }

    RealGenerator(RandT min, RandT max, unsigned int seed)
        : _dist(min, max), _seed(seed) {
        _rng.seed(_seed);
    }

    // Generator fun
    RandT generate_random() { return _dist(_rng); }

  private:
    // Type of random number distribution
    std::uniform_real_distribution<RandT> _dist;

    // Mersenne Twister: Good quality random number generator
    std::mt19937 _rng;

    // Seed
    unsigned int _seed;
};

template <typename RandT>
class IntGenerator {
  public:
    // Constructors
    IntGenerator(RandT min, RandT max) : _dist(min, max) {
        std::random_device rd;
        _seed = rd();
        _rng.seed(_seed);
    }

    IntGenerator(RandT min, RandT max, unsigned int seed)
        : _dist(min, max), _seed(seed) {
        _rng.seed(_seed);
    }

    // Generator fun
    RandT generate_random() { return _dist(_rng); }

  private:
    // Type of random number distribution
    std::uniform_int_distribution<RandT> _dist;

    // Mersenne Twister: Good quality random number generator
    std::mt19937 _rng;

    // Seed
    unsigned int _seed;
};
}  // namespace random_generators
