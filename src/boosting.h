#ifndef BOOSTING_H_
#define BOOSTING_H_

#include <math.h>

#include <algorithm>
#include <vector>

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

class Boosting {
   public:
    struct Edge {
        int node;
        double p;
        double pp;  // boosted probability
        Edge(int node, double p, double pp) : node(node), p(p), pp(pp) {}
    };

    Boosting(const char *graph_file, const char *seed_file, int k);
    Boosting(const char *graph_file, const char *seed_file, double beta, int k);
    void InitRest(const char *graph_file, const char *seed_file);

    ~Boosting() {}

    // graph
    int num_nodes_;
    int num_edges_;
    std::vector<std::vector<Edge> > in_edges_;

    // seed
    int num_seeds_;
    std::vector<bool> is_seed_;

    // k
    int k_;  // about boost maximization
    double beta_;

    // boost probability
    double BoostProb(double p) const { return 1.0 - std::pow(1 - p, beta_); }

    // constants
    static const double kEPS;

    // helper functions
    double ComputeSeedInf(int num_MC = 20000) const;
    double ComputeBoostMC(const std::vector<int> &boost,
                          int num_MC = 20000) const;
    static void PrintSolution(const char *const message, double influence_boost,
                              const std::vector<int> &boost);

   protected:
    // Use this to init seeds, if reproducibility is needed
    size_t seed_init_hash_;
};

#endif
