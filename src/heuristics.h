#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "boosting.h"
#include "getRSS.h"
#include "timer.h"

using std::set;
using std::string;
using std::vector;
using std::queue;
using std::pair;
using std::priority_queue;
using std::make_pair;

class Heuristics : public Boosting {
   public:
    Timer timer_;
    vector<vector<Edge>> out_edges_;
    vector<int> seeds_;
    size_t seed_init_hash_;

    void InitRest() {
        // Initialize out_edges_ and seeds_
        out_edges_.resize(num_nodes_);
        seeds_.clear();
        for (int i = 0; i < num_nodes_; i++) {
            for (const auto e : in_edges_[i]) {
                // i <- e.node
                out_edges_[e.node].emplace_back(i, e.p, e.pp);
            }
            if (is_seed_[i]) {
                seeds_.push_back(i);
            }
        }
    }

    Heuristics(const char *const graph_filename,
               const char *const seed_filename, double beta, int k)
        : Boosting(graph_filename, seed_filename, beta, k) {
        InitRest();
    }

    ~Heuristics() {
    }

    // Select nodes with top-k scores
    // Warning: score will be modified
    void NodeSelection(vector<double> &score, vector<int> &boost) {
        boost.resize(k_);
        for (int i = 0; i < k_; i++) {
            auto id = max_element(score.begin(), score.end());
            boost[i] = id - score.begin();
            (*id) = -1.0;
        }
    }

    void HighDegreeGlobal(vector<int> &boost, bool out = true,
                          bool discount = false) {
        // according to out_degree
        vector<double> sum_weight(num_nodes_, 0);
        // weight, node
        priority_queue<pair<double, int>> pq;
        for (int i = 0; i < num_nodes_; i++) {
            if (is_seed_[i]) {
                sum_weight[i] = -1.0;
            } else {
                sum_weight[i] = 0.0;
                for (const auto &e : (out ? out_edges_ : in_edges_)[i]) {
                    sum_weight[i] += out ? e.p : (e.pp - e.p);
                }
                pq.emplace(sum_weight[i], i);
            }
        }

        for (int i = 0; i < k_; i++) {
            double weight = pq.top().first;
            int best_node = pq.top().second;
            pq.pop();
            while (weight > sum_weight[best_node]) {
                // outdated weight in the priority queue
                pq.emplace(sum_weight[best_node], best_node);
                weight = pq.top().first;
                best_node = pq.top().second;
                pq.pop();
            }

            boost[i] = best_node;
            sum_weight[best_node] = -1;
            if (discount) {
                for (const auto &e :
                     (out ? in_edges_ : out_edges_)[best_node]) {
                    sum_weight[e.node] -= out ? e.p : (e.pp - e.p);
                }
            }
        }
    }

    // High Degree Local
    void HighDegreeLocal(vector<int> &boost, bool out = true,
                         bool discount = false) {
        vector<int> dis_from_seed(num_nodes_, num_edges_ + 1);
        queue<int> q;

        // Compute distance from seeds
        for (auto s : seeds_) {
            q.push(s);
            dis_from_seed[s] = 0;
        }
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (const auto &e : out_edges_[u]) {
                int v = e.node;
                if (dis_from_seed[u] + 1 < dis_from_seed[v]) {
                    dis_from_seed[v] = dis_from_seed[u] + 1;
                    q.push(v);
                }
            }
        }

        // Computed weighted degree
        vector<double> sum_weight(num_nodes_, 0);
        // <-dis, weight>, node
        priority_queue<pair<pair<int, double>, int>> pq;
        for (int i = 0; i < num_nodes_; i++) {
            if (is_seed_[i]) {
                sum_weight[i] = -1.0;
            } else {
                sum_weight[i] = 0.0;
                for (const auto &e : (out ? out_edges_ : in_edges_)[i]) {
                    sum_weight[i] += out ? e.p : (e.pp - e.p);
                }
                pq.emplace(make_pair(-dis_from_seed[i], sum_weight[i]), i);
            }
        }

        boost.resize(k_);
        for (int i = 0; i < k_; i++) {
            // int dis = -pq.top().first.first;
            double weight = pq.top().first.second;
            int best_node = pq.top().second;
            pq.pop();
            while (weight > sum_weight[best_node]) {
                // outdated weight in the priority queue
                pq.emplace(
                    make_pair(-dis_from_seed[best_node], sum_weight[best_node]),
                    best_node);
                // dis = -pq.top().first.first;
                weight = pq.top().first.second;
                best_node = pq.top().second;
                pq.pop();
            }

            boost[i] = best_node;
            sum_weight[best_node] = -1;
            if (discount) {
                for (const auto &e :
                     (out ? in_edges_ : out_edges_)[best_node]) {
                    sum_weight[e.node] -= out ? e.p : (e.pp - e.p);
                }
            }
        }
    }

    void PageRank(vector<int> &boost) {
        // Constants
        const double d = 0.85;
        const double stop_eps = 1e-4;

        // Prepare
        vector<double> sum_in_prob(num_nodes_, 0.0);
        for (int u = 0; u < num_nodes_; u++) {
            for (const auto &e : in_edges_[u]) {
                sum_in_prob[u] += e.p;
            }
        }
        vector<double> pr_old(num_nodes_, (1 - d) / num_nodes_);
        vector<double> pr_new(num_nodes_);

        // PageRank
        while (true) {
            double l1_norm_err = 0.0;
#pragma omp parallel for schedule(static, 1), reduction(+ : l1_norm_err)
            for (int u = 0; u < num_nodes_; u++) {
                pr_new[u] = 0.0;
                for (const auto e : out_edges_[u]) {
                    int v = e.node;
                    pr_new[u] += e.p / sum_in_prob[v] * pr_old[v];
                }
                pr_new[u] = d * pr_new[u] + (1 - d) / num_nodes_;
                l1_norm_err += fabs(pr_new[u] - pr_old[u]);
            }
            if (l1_norm_err < stop_eps)
                break;
            pr_new.swap(pr_old);
        }

        NodeSelection(pr_new, boost);
    }

    // Append logs to a log file.
    void PrintLogToFile(const char *log_filename,
                        const char *const graph_filename,
                        const vector<double> &boost_inf) {
        if (log_filename == NULL)
            return;

        FILE *f_log = fopen(log_filename, "a");
        if (f_log == NULL) {
            fprintf(stderr, "Cannot open %s as output file.\n", log_filename);
        }
        assert(f_log != NULL);

        fprintf(f_log, "%s\t%d\t%.0lf\t%d\t", graph_filename, num_seeds_, beta_,
                k_);
        for (auto b : boost_inf) {
            fprintf(f_log, "%.2lf\t", b);
        }
        fprintf(f_log, "\%.2lf\n", timer_.TimeElapsed());

        fclose(f_log);
    }
};
