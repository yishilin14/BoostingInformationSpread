#ifndef RRSET_H_
#define RRSET_H_

#include <cstdlib>

#include <deque>
#include <list>
#include <random>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "boosting.h"
#include "dsfmt/dSFMT.h"

class RRSet : public Boosting {
   public:
    RRSet(const char *graph_file, const char *seed_file, double beta, int k);
    ~RRSet();

    // Generate rr sets
    void InsertRR(int count);

#ifndef LBONLY
    // Select a set of nodes according to the true boost
    // If "before_bookmark" is ture,
    // RR-sets generated after(>=) "the bookmark" will be ignored.
    void NodeSelectionTrue(std::vector<int> &boost, double &boost_true_score,
                           bool before_bookmark = false) const;

    // Compute boost (ignoring old RR-sets are allowed)
    // If "after_bookmark" is ture,
    // RR-sets generated before(<) "the bookmark" will be ignored.
    double ComputeBoost(const std::vector<int> &boost,
                        bool after_bookmark = false) const;

    void CheckCriticalNodes(const std::vector<int> &boost) const;

    // Test the tightness of lower bound
    void LowerBoundTest(int test_size, const char *lb_log_filename,
                        const char *grpah_file, double epsilon);
#endif

    // void NodeSelectionLowerBound(std::vector<int> &boost,
    //                              double &boost_lb_true_score,
    //                              double &boost_lb_lb_score) const;
    void NodeSelectionLowerBound(std::vector<int> &boost,
                                 double &boost_lb_lb_score) const;

    // Compute the lower bound of the boost
    // If "after_bookmark" is ture,
    // RR-sets generated before(<) "the bookmark" will be ignored.
    double ComputeBoostLowerBound(const std::vector<int> &boost,
                                  bool after_bookmark = false) const;
    double ComputeBoostLowerBound(const std::vector<bool> &is_boost,
                                  bool after_bookmark = false) const;

    // Recompute boost by generating a set of new RR-sets
    double ReComputeBoost(const std::vector<int> &boost, int num_RR);

    // Print statistics to screen
    void PrintStatistics() const;

   protected:
    typedef int DistanceType;
    static const DistanceType kMaxDis;

    static const int kSuperSeed;
    static const int kInvalidNodePosition;

    enum RREdgeType {
        kActive = 0,
        kBoost,
    };
    class RREdge {
       private:
        unsigned int edge_{0};  // the last bit is the edge type
        int next_{-1};

       public:
        RREdge() {}
        RREdge(int node, RREdgeType type)
            : edge_((((unsigned int)node) << 1) | (type == kActive ? 1 : 0)) {}
        RREdge(int node, RREdgeType type, int next)
            : edge_((((unsigned int)node) << 1) | (type == kActive ? 1 : 0)),
              next_(next) {}
        int node() const { return edge_ >> 1; }
        RREdgeType type() const { return (edge_ & 1) ? kActive : kBoost; }
        DistanceType distance() const { return (edge_ & 1) ^ 1; }
        bool IsActive() const { return edge_ & 1; }
        int get_next() const { return next_; }
        void reset(int node, RREdgeType type) {
            edge_ = (((unsigned int)node) << 1) | (type == kActive ? 1 : 0);
        }
    };

    // Global statistics (should be consistent with the statistics in RRBags)
    int num_RR_;
    int num_RR_no_seed_;
    int num_RR_activated_;
    int num_RR_boostable_;
    int num_RR_has_critical_;

    double num_nodes_to_boost_;
    double num_critical_nodes_;

    double sum_nodes_;
    double sum_edges_;
    double sum_active_edges_;
    double sum_uncompressed_nodes_;
    double sum_uncompressed_edges_;

    void InitRRBags();
    void InitNewRRHelpers();
    void ResetGlobalStastics();
    void UpdateGlobalStatistics();
    int NewRR(int tid);
    void CompressRR(int root, int tid);
    void InsertRRBookmark();

    // Some helper data structures
   private:
    struct RRBag {
        // RR sets
        int num_RR;

        std::vector<int>
            RR_heads;  // position of the first node of every RR set
        std::vector<int> RR_nodes;  // nodes of RR sets
        std::vector<std::vector<int>> RR_critical;  // critical nodes of RR sets

        std::vector<int> first_parent;
        std::vector<RREdge> parents;

        std::vector<int> first_child;
        std::vector<RREdge> children;

        std::vector<std::vector<int>> nodes_boost_RR;  // nodes -> RR id
        std::vector<std::vector<int>> nodes_activate_RR;  // critical nodes -> RR id

        std::vector<double>
            score_init;  // initial score (when B == emptyset)

        int num_RR_no_seed;
        int num_RR_activated;
        int num_RR_boostable;
        int num_RR_has_critical;

        double num_nodes_to_boost;
        double num_critical_nodes;

        double sum_nodes;
        double sum_edges;
        double sum_active_edges;
        double sum_uncompressed_nodes;
        double sum_uncompressed_edges;

        int bookmark_num_RR;
        int bookmark_num_RR_boostable;
    };
    RRBag rrbag_[NUM_THREADS];

    struct NewRRHelper {
        dsfmt_t dsfmt_seed;
        int visit_id;
        std::vector<int> visited;
        std::vector<DistanceType> dis_to_root;
        std::vector<DistanceType> dis_from_seeds;
        std::vector<std::vector<RREdge>> parents;
        std::vector<std::vector<RREdge>> children;
        std::vector<int> node_pos;
        std::deque<std::pair<int, DistanceType>> dq1;  // for generating RR set
        std::deque<std::pair<int, DistanceType>> dq2;  // for computing
    };
    NewRRHelper genrr_[NUM_THREADS];
};

#endif
