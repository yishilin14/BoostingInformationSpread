#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <set>
#include <string>
#include <vector>

#include "getRSS.h"
#include "rrset.h"
#include "timer.h"

using std::set;
using std::string;
using std::vector;

const char *const kSeperateLine =
    "\n\n==================================================\n";

class BoostIC : public RRSet {
   public:
    double l_;
    double epsilon_;
    double lambda_star_;

    Timer timer_;
    double total_time_;
    double peak_mem_{0};
    double init_mem_{0};

    double boost_best_{0};
    double boost_lb_lb_{0};
    double boost_lb_true_{0};
    double boost_true_{0};
    double boost_recompute_{0};

    vector<int> boost_set_;
    vector<int> boost_set_lb_;

    BoostIC(const char *const graph_filename, const char *const seed_filename,
            double beta, int k, double epsilon)
        : RRSet(graph_filename, seed_filename, beta, k),
          l_(1.0 + log(3) / log(num_nodes_)),
          epsilon_(epsilon) {
        if (k_ >= num_nodes_) {
            fprintf(stderr, "k_ >= num_nodes_, boost all nodes...\n");
            exit(-1);
        }
        double log_n_choose_k = num_nodes_ * log(num_nodes_) - k_ * log(k_) -
                                (num_nodes_ - k_) * log(num_nodes_ - k_);
        double aa = sqrt(l_ * log(num_nodes_) + log(2));
        double bb = sqrt((1 - 1 / exp(1)) *
                         (log_n_choose_k + l_ * log(num_nodes_) + log(2)));
        lambda_star_ = 2 * num_nodes_ * pow((1 - 1 / exp(1)) * aa + bb, 2) /
                       (epsilon_ * epsilon_);
    }

    ~BoostIC() {}

    int ComputeTheta(double lower_bound) { return lambda_star_ / lower_bound; }

    void Sampling() {
        vector<int> boost(k_);

        double lb = 1;
        double eps_prime = sqrt(2) * epsilon_;
        double log_n_choose_k = num_nodes_ * log(num_nodes_) - k_ * log(k_) -
                                (num_nodes_ - k_) * log(num_nodes_ - k_);
        double boost_lb_lb;

        for (int i = 1; i < log(num_nodes_) / log(2); i++) {
            // Compute theta_i.
            double x = num_nodes_ / pow(2, i);
            double lambda_prime =
                (2 + 2 / 3 * eps_prime) *
                (log_n_choose_k + l_ * log(num_nodes_) + log(log(num_nodes_))) *
                num_nodes_;
            lambda_prime /= (eps_prime * eps_prime);
            int theta_i = (int)(lambda_prime / x);

            puts(kSeperateLine);
            printf("[BoostIC::Sampling] i = %d, theta_i = %d\n", i, theta_i);

            // Generate new RR sets.
            InsertRR(theta_i - num_RR_);

            // Print messages.
            printf("[BoostIC::Sampling] Time elapsed = %.2lf (s).\n",
                   timer_.TimeElapsed());
            printf("[BoostIC::Sampling] Peak memory usage = %.2lf (GB)\n",
                   getPeakRSSGB());

            // Select nodes according to the lower bound function.
            NodeSelectionLowerBound(boost, boost_lb_lb);
            if (boost_lb_lb >= (1 + eps_prime) * x) {
                lb = boost_lb_lb / (1 + eps_prime);
                break;
            }
        }

        // Compute theta.
        int theta = ComputeTheta(lb);
        puts(kSeperateLine);
        printf("[BoostIC::Sampling] theta = %d\n", theta);

        // Generate RR sets.
        InsertRR(theta - num_RR_);

        // Print information about Sampling().
        printf("[BoostIC::Sampling] Time elapsed = %.2lf (s).\n",
               timer_.TimeElapsed());
        printf("[BoostIC::Sampling] Peak memory usage = %.2lf (GB)\n",
               getPeakRSSGB());
    }

    // Sample and select nodes.
    void Solve() {
        init_mem_ = getPeakRSSGB();
        printf("[BoostIC::Solve] Time elapsed = %.2lf (s).\n", total_time_);
        printf("[BoostIC::Solve] Peak memory usage = %.2lf (GB)\n", init_mem_);

        /// Sampling
        Sampling();
        puts(kSeperateLine);

        // Use lower bound
        NodeSelectionLowerBound(boost_set_lb_, boost_lb_lb_);
        PrintSolution("Node selected by the lower bound", boost_lb_lb_,
                      boost_set_lb_);

// Sandwich
#ifdef LBONLY
        // solution of sandwich = lb
        boost_set_ = boost_set_lb_;
        boost_lb_true_ = boost_lb_lb_;
        boost_best_ = boost_lb_true_;
#else
        NodeSelectionTrue(boost_set_, boost_true_);
        PrintSolution("Node selected by the true boost", boost_true_,
                      boost_set_);
        boost_lb_true_ = ComputeBoost(boost_set_lb_);
        if (boost_true_ > boost_lb_true_) {
            boost_best_ = boost_true_;
        } else {
            boost_set_ = boost_set_lb_;
            boost_best_ = boost_lb_true_;
        }
#endif

        // Print results and return
        printf("Final boost = %.2lf\n", boost_best_);
        total_time_ = timer_.TimeElapsed();
        peak_mem_ = getPeakRSSGB();

        // Print logs to screen.
        puts(kSeperateLine);
        PrintStatistics();
        puts(kSeperateLine);
        printf("[BoostIC::Solve] Time elapsed = %.2lf (s).\n", total_time_);
        printf("[BoostIC::Solve] Peak memory usage = %.2lf (GB)\n", peak_mem_);

#ifndef LBONLY
        puts(kSeperateLine);
        CheckCriticalNodes(boost_set_);
#endif
    }

    void Recompute() {
#ifdef RECOMPUTE
        // Recompute the boost
        puts(kSeperateLine);
        boost_recompute_ = ComputeBoostMC(boost_set_);
        printf("[BoostIC::Recompute] Time elapsed = %.2lf (s).\n",
               timer_.TimeElapsed());
#endif
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
        fprintf(f_log, "%.2lf\t", (double)num_RR_no_seed_ / num_RR_);
        fprintf(f_log, "%.2lf\t", (double)num_RR_activated_ / num_RR_);
        fprintf(f_log, "%.2lf\t", (double)num_RR_boostable_ / num_RR_);

        // About boostable RR sets
        fprintf(f_log, "%.2lf\t", sum_nodes_ / num_RR_boostable_);
        fprintf(f_log, "%.2lf\t", sum_edges_ / num_RR_boostable_);
        fprintf(f_log, "%.2lf\t", sum_uncompressed_nodes_ / num_RR_boostable_);
        fprintf(f_log, "%.2lf\t", sum_uncompressed_edges_ / num_RR_boostable_);
        fprintf(f_log, "%.2lf\t", sum_active_edges_ / num_RR_boostable_);
        fprintf(f_log, "%.2lf\t",
                (double)num_nodes_to_boost_ / num_RR_boostable_);
        fprintf(f_log, "%.2lf\t",
                (double)num_RR_has_critical_ / num_RR_boostable_);
        fprintf(f_log, "%.2lf\t",
                (double)num_critical_nodes_ / num_RR_has_critical_);

#ifdef LBONLY
        // Node selection results
        fprintf(f_log, "0.00\t0.00\t0.00\t0.00\t");
#else
        // Node selection results
        fprintf(f_log, "%.2lf\t", boost_true_);
        fprintf(f_log, "%.2lf\t%.2lf\t", boost_lb_true_,
                boost_lb_lb_ / boost_lb_true_);
        fprintf(f_log, "%.2lf\t", boost_best_);
#endif

        fprintf(f_log, "%.2lf\t%.2lf\t", total_time_, peak_mem_);
        fprintf(f_log, "%.2lf\t", boost_recompute_);
        fprintf(f_log, "%.2lf\t", init_mem_);

        fprintf(f_log, "\n");

        fclose(f_log);
    }
};

// Parameters: graph_nm seed_filename k epsilon log_file
int main(int argc, const char **argv) {
// Check parameters
#ifdef TESTLB
    if (argc < 8) {
        printf(
            "Wrong parameters! (graph_filename, seed_filename, beta, k, "
            "epsilon, log_filename, log_lb_filename)\n");
        exit(-1);
    }
#else
    if (argc < 6) {
        printf(
            "Wrong parameters! (graph_filename, seed_filename, beta, k, "
            "epsilon, log_filename(optional)\n");
        exit(-1);
    }
#endif

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
    BoostIC boostic = BoostIC(graph_filename, seed_filename, beta, k, epsilon);
    boostic.Solve();

    // Recompute the boost of influence
    boostic.Recompute();

    // Print logs to file
    boostic.PrintLogToFile(log_filename, graph_filename);

#ifdef TESTLB
    const char *log_lb_filename = argv[7];
    boostic.LowerBoundTest(300, log_lb_filename, graph_filename, epsilon);
#endif

    puts(kSeperateLine);
    Timer::PrintCurrentTime("All finished at");
    return 0;
}
