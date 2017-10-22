#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <queue>
#include <random>
#include <vector>

#include <omp.h>

#include "boosting.h"
#include "dsfmt/dSFMT.h"

using std::vector;
using std::queue;
using std::priority_queue;

const double Boosting::kEPS = 1e-8;

// Read the graph
Boosting::Boosting(const char *graph_file, const char *seed_file, int k)
    : k_(k), beta_(2.0) {
    InitRest(graph_file, seed_file);
}

Boosting::Boosting(const char *graph_file, const char *seed_file, double beta,
                   int k)
    : k_(k), beta_(beta) {
    InitRest(graph_file, seed_file);
}

// Read the graph
void Boosting::InitRest(const char *graph_file, const char *seed_file) {
    // Read the graph
    FILE *f_graph = fopen(graph_file, "r");
    if (f_graph == NULL) {
        fprintf(stderr, "Cannot open %s!\n", graph_file);
        exit(-1);
    }
    if (fscanf(f_graph, "%d%d", &num_nodes_, &num_edges_) != 2) {
        fprintf(stderr, "Wrong graph format!\n");
        exit(-1);
    }

    in_edges_.resize(num_nodes_);
    for (int i = 0; i < num_nodes_; i++) {
        in_edges_[i].clear();
    }
    for (int i = 0; i < num_edges_; i++) {
        int u, v;
        double p;
        if (fscanf(f_graph, "%d%d%lf", &u, &v, &p) != 3) {
            fprintf(stderr, "Wrong graph format!\n");
            exit(-1);
        }
        if (u == v) {
            fprintf(stderr, "Remove self-loops first!\n");
            exit(-1);
        }
        in_edges_[v].emplace_back(u, p, BoostProb(p));
    }
    fclose(f_graph);

    // Read the seed set
    FILE *f_seed = fopen(seed_file, "r");
    if (f_seed == NULL) {
        fprintf(stderr, "Cannot open %s!\n", seed_file);
        exit(-1);
    }
    is_seed_.resize(num_nodes_);
    std::fill(is_seed_.begin(), is_seed_.end(), false);
    num_seeds_ = 0;
    int crt_seed;
    while (fscanf(f_seed, "%d", &crt_seed) == 1) {
        is_seed_[crt_seed] = true;
        num_seeds_++;
    }
    fclose(f_seed);

    printf("[Boosting::Boosting] %d nodes, %d edges, %d seeds\n", num_nodes_,
           num_edges_, num_seeds_);

    seed_init_hash_ =
        std::hash<double>()(1.0 * num_nodes_ * num_edges_ * num_seeds_ * k_);

    // Init omp
    omp_set_num_threads(NUM_THREADS);
}

double Boosting::ComputeSeedInf(int num_MC) const {
    vector<vector<Edge> > out_edges_(num_nodes_);
    vector<int> seeds_;

    // Initialize out_edges_ and seeds_
    out_edges_.resize(num_nodes_);
    seeds_.clear();
    for (int i = 0; i < num_nodes_; i++) {
        for (const auto e : in_edges_[i])
            out_edges_[e.node].emplace_back(i, e.p, e.pp);
        if (is_seed_[i]) seeds_.push_back(i);
    }

    double cnt_before{0.0};

#pragma omp parallel
    {
        // Reproducible
        dsfmt_t dsfmt_seed;
        dsfmt_init_gen_rand(&dsfmt_seed,
                            seed_init_hash_ + omp_get_thread_num());

        // For MC
        vector<int> q(num_nodes_);  // queue
        vector<int> visit(num_nodes_, -1);

#pragma omp for schedule(static, 1), reduction(+ : cnt_before)
        for (int mc = 0; mc < num_MC; mc++) {
            int head{0}, tail{0};

            int cnt = 0;
            /// Compute the original influence
            for (const auto s : seeds_) {
                q[tail++] = s;
                visit[s] = mc;
            }
            while (head < tail) {
                int u = q[head++];
                // Sample edge type
                for (auto e : out_edges_[u]) {
                    if (visit[e.node] == mc) continue;  // visited
                    double r = dsfmt_genrand_open_close(&dsfmt_seed);
                    if (r <= e.p) {
                        q[tail++] = e.node;
                        visit[e.node] = mc;
                        cnt++;
                    }
                }
            }
            cnt_before += cnt;
        }  // end of for
    }      // end of omp parallel block

    double influence = num_seeds_ + (cnt_before / num_MC);
    printf("[Boosting::ComputeSeedInf] inf = %.2lf\n", influence);
    return influence;
}

double Boosting::ComputeBoostMC(const vector<int> &boost, int num_MC) const {
    vector<vector<Edge> > out_edges_(num_nodes_);
    vector<int> seeds_;

    // Initialize out_edges_ and seeds_
    out_edges_.resize(num_nodes_);
    seeds_.clear();
    for (int i = 0; i < num_nodes_; i++) {
        for (const auto e : in_edges_[i])
            out_edges_[e.node].emplace_back(i, e.p, e.pp);
        if (is_seed_[i]) seeds_.push_back(i);
    }

    vector<bool> is_boost(num_nodes_, false);
    for (auto b : boost) {
        is_boost[b] = true;
    }

    double cnt_before{0.0};
    double cnt_after{0.0};

#pragma omp parallel
    {
        // Reproducible
        dsfmt_t dsfmt_seed;
        dsfmt_init_gen_rand(&dsfmt_seed,
                            seed_init_hash_ + omp_get_thread_num());

        // For MC
        vector<int> q(num_nodes_);        // queue
        vector<int> q_boost(num_nodes_);  // queue
        vector<int> visit(num_nodes_, -1);

#pragma omp for schedule(static, 1), reduction(+ : cnt_before, cnt_after)
        for (int mc = 0; mc < num_MC; mc++) {
            int head{0}, tail{0};
            int boost_head{0}, boost_tail{0};

            int cnt = 0;
            /// Compute the original influence
            for (const auto s : seeds_) {
                q[tail++] = s;
                visit[s] = mc;
            }
            while (head < tail) {
                int u = q[head++];
                // Sample edge type
                for (auto e : out_edges_[u]) {
                    if (visit[e.node] == mc) continue;  // visited
                    double r = dsfmt_genrand_open_close(&dsfmt_seed);
                    if (r <= e.p) {
                        q[tail++] = e.node;
                        visit[e.node] = mc;
                        cnt++;
                    } else if (r <= e.pp && is_boost[e.node]) {
                        q_boost[boost_tail++] = e.node;
                    }
                }
            }
            cnt_before += cnt;

            while (boost_head < boost_tail) {
                int u = q_boost[boost_head++];
                if (visit[u] == mc) continue;  // active w/o boost
                // Sample edge type
                for (auto e : out_edges_[u]) {
                    if (visit[e.node] == mc || visit[e.node] == mc + num_MC)
                        continue;  // visited
                    double r = dsfmt_genrand_open_close(&dsfmt_seed);
                    if (r <= e.p || (r <= e.pp && is_boost[e.node])) {
                        q_boost[boost_tail++] = e.node;
                        visit[e.node] = mc + num_MC;
                        cnt++;
                    }
                }
            }
            cnt_after += cnt;
        }  // end of for
    }      // end of omp parallel block

    double inf_boost = (cnt_after - cnt_before) / num_MC;
    printf("[Boosting::ComputeBoost] inf = %.2lf -> %.2lf, boost = %.2lf\n",
           num_seeds_ + (cnt_before / num_MC),
           num_seeds_ + (cnt_after / num_MC), inf_boost);
    return inf_boost;
}

void Boosting::PrintSolution(const char *const message, double influence_boost,
                             const vector<int> &boost) {
    printf("\n[Boosting::PrintSolution (%s)]\n", message);
    printf("Boosted influence: %.2lf\n", influence_boost);
    printf("Boosted nodes: ");
    int cnt = 0;
    for (auto b : boost) {
        printf("%d ", b);
        cnt++;
        if (cnt == 50) {
            printf("...");
            break;
        }
    }
    printf("\n\n");
}
