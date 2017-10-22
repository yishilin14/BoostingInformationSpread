#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "boosting.h"
#include "dsfmt/dSFMT.h"
#include "getRSS.h"
#include "timer.h"

using std::set;
using std::string;
using std::vector;
using std::queue;

const char *const kSeperateLine =
    "\n\n==================================================\n";

class MoreSeeds : public Boosting {
   public:
    double delta_;
    double epsilon_;

    Timer timer_;
    double total_time_;
    double peak_mem_{0};
    double init_mem_{0};

    // Results
    vector<int> solution_;
    double inc_inf_;
    double boost_;

    // RR sets
    int num_RR_;
    int num_useful_RR_;  // not initially activated
    vector<vector<int>> RR_has_node_;
    vector<vector<int>> node_to_RR_;

    struct NewRRHelper {
        dsfmt_t dsfmt_seed;
        int visit_id;
        vector<int> visited;
        vector<int> q;
    };
    NewRRHelper helper;

    MoreSeeds(const char *const graph_filename, const char *const seed_filename,
              double beta, int k, double epsilon)
        : Boosting(graph_filename, seed_filename, beta, k),
          delta_(1.0 / 1000),
          epsilon_(epsilon) {
        if (k_ >= num_nodes_) {
            fprintf(stderr, "k_ >= num_nodes_, boost all nodes...\n");
            exit(-1);
        }

        // Initialize solution
        solution_.resize(k_);
        inc_inf_ = 0.0;

        // Initialize RR sets
        num_RR_ = 0;
        num_useful_RR_ = 0;
        RR_has_node_.clear();
        node_to_RR_.resize(num_nodes_);

        // Initialize helper
        dsfmt_init_gen_rand(&helper.dsfmt_seed, time(NULL));
        helper.visit_id = 0;
        helper.visited.resize(num_nodes_, 0);
        helper.q.resize(num_nodes_);

        printf("Seeds:");
        for (int i = 0; i < num_nodes_; i++) {
            if (is_seed_[i]) printf("%d ", i);
        }
        printf("\n");
    }

    ~MoreSeeds() {}

    bool CheckStop(double inc_inf, double check_inc_inf, int num_RR_old,
                   double delta_ssa) const {
        const static double E = exp(1.0);
        static double c = 2.0 * (E - 2.0);
        double eps1 = inc_inf / check_inc_inf - 1;
        printf("[MoreSeeds::CheckStop] eps1: %.2lf\n", eps1);
        if (eps1 >= 0 && eps1 <= epsilon_) {
            double eps2 = (epsilon_ - eps1) / 2 / (1 + eps1);
            double eps3 = (epsilon_ - eps1) / 2 / (1 - 1 / E);
            double cover = inc_inf * num_RR_old / num_nodes_;
            double check_cover = check_inc_inf * num_RR_old / num_nodes_;
            double delta1 =
                exp(-(cover * pow(eps3, 2) / 2 / c / (1 + eps1) / (1 + eps2)));
            double delta2 =
                exp(-((check_cover - 1) * pow(eps2, 2) / 2 / c / (1 + eps2)));
            printf(
                "[MoreSeeds::CheckStop] delta: %.8lf, %.8lf, (%.8lf,%.8lf)\n",
                delta1, delta2, delta1 + delta2, delta_ssa);
            if (delta1 + delta2 <= delta_ssa) {
                printf("[MoreSeeds::CheckStop] OK\n");
                return true;
            }
        }
        printf("[MoreSeeds::CheckStop] Failed\n");
        return false;
    }

    void newRR() {
        // Randomly select "root"
        int root = dsfmt_genrand_uint32(&helper.dsfmt_seed) % num_nodes_;
        if (is_seed_[root]) {
            return;  // already activated
        }

        // Initialize temporary RR set
        helper.visit_id++;  // assume helper.visit_id < INT_MAX
        helper.visited[root] = helper.visit_id;

        // Generate RR set (and compute helper.dis_to_root)
        // - Return false immediately if this RR set is activated
        int head = 0, tail = 0;
        helper.q[tail++] = root;
        while (head < tail) {
            int u = helper.q[head++];

            // For each incoming edge of u, ...
            for (const auto &e : in_edges_[u]) {
                int v = e.node;  // v->u
                if (helper.visited[v] == helper.visit_id) continue;
                // Check whether if we should consider this edge
                double r = dsfmt_genrand_open_close(&helper.dsfmt_seed);
                if (r > e.p) continue;
                if (is_seed_[v]) return;  // activated
                // Update the queue
                helper.visited[v] = helper.visit_id;
                helper.q[tail++] = v;
            }
        }

        // Finalize RR set
        RR_has_node_.push_back(
            vector<int>(helper.q.begin(), helper.q.begin() + tail));
        for (int i = 0; i < tail; i++) {
            node_to_RR_[helper.q[i]].push_back(num_useful_RR_);
        }
        num_useful_RR_++;
    }

    void InsertRR(int num_new_RR) {
        for (int i = 0; i < num_new_RR; i++) {
            newRR();
        }
        num_RR_ += num_new_RR;
        printf("[MoreSeeds::InsertRR] num_RR_ = %d, num_useful_RR_ = %d\n",
               num_RR_, num_useful_RR_);
        Timer::PrintCurrentTime("[MoreSeeds::InsertRR] finished at ");
    }

    void NodeSelection() {
        printf("[MoreSeeds::NodeSelection] ");
        vector<double> score(num_nodes_);
        vector<bool> covered(num_useful_RR_, false);
        for (int i = 0; i < num_nodes_; i++) {
            score[i] = node_to_RR_[i].size();
        }

        inc_inf_ = 0.0;
        for (int i = 0; i < k_; i++) {
            int best_node = -1;
            for (int j = 0; j < num_nodes_; j++) {
                if (score[j] < 0) continue;
                if (best_node == -1 || score[j] > score[best_node]) {
                    best_node = j;
                }
            }
            if (best_node == -1) {
                best_node = 0;  // in case there is no enough nodes to choose
            }
            solution_[i] = best_node;
            inc_inf_ += score[best_node];
            // Update scores
            for (auto RR_id : node_to_RR_[best_node]) {
                if (covered[RR_id]) continue;
                covered[RR_id] = true;
                for (auto u : RR_has_node_[RR_id]) score[u]--;
            }
            printf("%d ", solution_[i]);
        }
        inc_inf_ = inc_inf_ * num_nodes_ / num_RR_;
        printf("\nIncrease of influence: %.2lf\n", inc_inf_);

        // double test = ComputeIncInf(0, 0);
        // if (test != inc_inf_) {
        //     printf("%.2lf %.2lf\n", inc_inf_, test);
        //     exit(-1);
        // }
    }

    double ComputeIncInf(int num_useful_RR_old, int num_RR_old) {
        vector<bool> covered(num_useful_RR_, false);
        double check_inc_inf = 0.0;
        for (int u : solution_) {
            // Update scores
            for (auto RR_id : node_to_RR_[u]) {
                if (RR_id < num_useful_RR_old) continue;
                if (covered[RR_id]) continue;
                covered[RR_id] = true;
                check_inc_inf++;
            }
        }
        return check_inc_inf * num_nodes_ / (num_RR_ - num_RR_old);
    }

    // Sample and select nodes.
    void Solve() {
        init_mem_ = getPeakRSSGB();
        printf("[MoreSeeds::Solve] Time elapsed = %.2lf (s).\n", total_time_);
        printf("[MoreSeeds::Solve] Peak memory usage = %.2lf (GB)\n",
               init_mem_);

        // D-SSA (SIGMOD'16)
        double c = 2.0 * (exp(1.0) - 2.0);
        double delta_ssa = delta_ / 2.0;
        int Lambda = 2 * c * pow(1 + epsilon_, 2) *
                     log(2.0 / delta_ssa / pow(epsilon_, 2));

        puts(kSeperateLine);
        printf("[MoreSeeds::D-SSA] eps_ssa: %.2lf\n", epsilon_);
        printf("[MoreSeeds::D-SSA] delta_ssa: %.2lf\n", delta_ssa);
        printf("[MoreSeeds::D-SSA] initial sample size: %d\n", Lambda);

        // D-SSA initialization: Generate initial RR sets
        InsertRR(Lambda);
        NodeSelection();

        // D-SSA initialization: Compute the upper bound for sampling size
        double log_n_choose_k = num_nodes_ * log(num_nodes_) - k_ * log(k_) -
                                (num_nodes_ - k_) * log(num_nodes_ - k_);
        double num_RR_upper_bound = (8 + 2 * epsilon_) * num_nodes_ *
                                    (log(2 / delta_ssa) + log_n_choose_k) /
                                    pow(epsilon_, 2);
        printf("[MoreSeeds::D-SSA] max num of RR sets = %.0lf\n",
               num_RR_upper_bound);

        // D-SSA: Main loop
        // Use D-SSA to select boost_set_lb_
        for (int itr = 1; num_RR_ < num_RR_upper_bound; itr++) {
            // Print logs
            printf("[MoreSeeds::D-SSA] itr = %d, sample size = %d\n", itr,
                   num_RR_ << 1);

            // Generate RR sets
            int num_RR_old = num_RR_;
            int num_useful_RR_old = num_useful_RR_;
            InsertRR(num_RR_);

            // Print messages.
            printf("[MoreSeeds::D-SSA] Time elapsed = %.2lf (s).\n",
                   timer_.TimeElapsed());
            printf("[MoreSeeds::D-SSA] Peak memory usage = %.2lf (GB)\n",
                   getPeakRSSGB());

            // Check whether we can stop
            double check_inc_inf = ComputeIncInf(num_useful_RR_old, num_RR_old);
            printf("[MoreSeeds::D-SSA] Check: %.2lf %.2lf\n", inc_inf_,
                   check_inc_inf);
            if (CheckStop(inc_inf_, check_inc_inf, num_RR_old, delta_ssa)) {
                break;
            }

            // Select nodes again
            NodeSelection();
        }

        // Print results
        printf("Final increment of influence = %.2lf\n", inc_inf_);
        total_time_ = timer_.TimeElapsed();
        peak_mem_ = getPeakRSSGB();

        // Compute boost of the influence spread
        boost_ = ComputeBoostMC(solution_);
        puts(kSeperateLine);
        printf("[MoreSeeds::Solve (Compute boost)] Time elapsed = %.2lf (s).\n",
               timer_.TimeElapsed());
        puts(kSeperateLine);

        // Print logs to screen.
        puts(kSeperateLine);
        printf("[MoreSeeds::Solve] Time elapsed = %.2lf (s).\n", total_time_);
        printf("[MoreSeeds::Solve] Peak memory usage = %.2lf (GB)\n",
               peak_mem_);
    }

    // Append logs to a log file.
    void PrintLogToFile(const char *log_filename,
                        const char *const graph_filename) {
        if (log_filename == NULL) return;

        FILE *f_log = fopen(log_filename, "a");
        if (f_log == NULL) {
            fprintf(stderr, "Cannot open %s as output file.\n", log_filename);
        }
        assert(f_log != NULL);

        // About datasets: #nodes, #edges, #seeds, k, epsilon
        fprintf(f_log, "%s\t", graph_filename);
        fprintf(f_log, "%d\t%d\t", num_nodes_, num_edges_);
        fprintf(f_log, "%d\t%.2lf\t%d\t%.2lf\t", num_seeds_, beta_, k_,
                epsilon_);

        // About RR sets
        fprintf(f_log, "%d\t", num_RR_);
        fprintf(f_log, "%d\t", num_useful_RR_);

        // About quality
        fprintf(f_log, "%.2lf\t", inc_inf_);
        fprintf(f_log, "%.2lf\t", boost_);

        // Time & memory
        fprintf(f_log, "%.2lf\t%.2lf\t", total_time_, peak_mem_);
        fprintf(f_log, "%.2lf\t", init_mem_);

        fprintf(f_log, "\n");
        fclose(f_log);
    }
};

// Parameters: graph_nm seed_filename k epsilon log_file
int main(int argc, const char **argv) {
    // Check parameters
    if (argc < 6) {
        printf(
            "Wrong parameters! (graph_filename, seed_filename, beta, k, "
            "epsilon, log_filename(optional)\n");
        exit(-1);
    }

    // Get and print parameters
    const char *graph_filename = argv[1];
    const char *seed_filename = argv[2];
    double beta = atof(argv[3]);
    int k = atoi(argv[4]);
    double epsilon = atof(argv[5]);
    const char *log_filename = argc > 6 ? argv[6] : NULL;
    srand(time(0));
    printf("[Parameters] %s %s %.2lf %d %.2lf\n", graph_filename, seed_filename,
           beta, k, epsilon);

    // PRR-Boost
    MoreSeeds more_seeds =
        MoreSeeds(graph_filename, seed_filename, beta, k, epsilon);
    more_seeds.Solve();

    // Print logs to file
    more_seeds.PrintLogToFile(log_filename, graph_filename);

    puts(kSeperateLine);
    Timer::PrintCurrentTime("All finished at");
    return 0;
}
