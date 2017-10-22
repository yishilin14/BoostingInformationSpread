#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <vector>

#include <omp.h>

#include "rrset.h"
#include "timer.h"

using std::vector;
using std::queue;
using std::deque;
using std::pair;
using std::set;
using std::stack;

const RRSet::DistanceType RRSet::kMaxDis =
    std::numeric_limits<RRSet::DistanceType>::max() >> 1;
const int RRSet::kSuperSeed = std::numeric_limits<int>::max();
const int RRSet::kInvalidNodePosition = -1;

Timer timer_global;

void RRSet::InitNewRRHelpers() {
    vector<int> odeg(num_nodes_, 0);
    for (int u = 0; u < num_nodes_; u++) {
        for (const auto e : in_edges_[u]) {
            odeg[e.node]++;
        }
    }

    // Initialize random number engine
    dsfmt_t seed;
    dsfmt_init_gen_rand(&seed, time(NULL));

    for (int tid = 0; tid < NUM_THREADS; tid++) {
        dsfmt_init_gen_rand(&genrr_[tid].dsfmt_seed,
                            dsfmt_genrand_uint32(&seed));

        // Reset vectors for generating rr-sets
        genrr_[tid].visit_id = 0;
        genrr_[tid].visited.resize(num_nodes_);
        // We won't use 0 as a valid id in the generation
        std::fill(genrr_[tid].visited.begin(), genrr_[tid].visited.end(), 0);
        genrr_[tid].dis_to_root.resize(num_nodes_);
        genrr_[tid].dis_from_seeds.resize(num_nodes_);
        genrr_[tid].node_pos.resize(num_nodes_);
        std::fill(genrr_[tid].node_pos.begin(), genrr_[tid].node_pos.end(),
                  kInvalidNodePosition);

        genrr_[tid].parents.resize(num_nodes_);
        genrr_[tid].children.resize(num_nodes_);
        for (int j = 0; j < num_nodes_; j++) {
            genrr_[tid].children[j].reserve(in_edges_[j].size());
            genrr_[tid].parents[j].reserve(odeg[j]);
        }
    }
}

void RRSet::InitRRBags() {
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        // Initialize global statistics
        rrbag_[tid].num_RR = 0;
        rrbag_[tid].num_RR_no_seed = 0;
        rrbag_[tid].num_RR_activated = 0;
        rrbag_[tid].num_RR_boostable = 0;
        rrbag_[tid].num_nodes_to_boost = 0;
        rrbag_[tid].num_RR_has_critical = 0;
        rrbag_[tid].num_critical_nodes = 0;
        rrbag_[tid].sum_nodes = 0;
        rrbag_[tid].sum_edges = 0;
        rrbag_[tid].sum_active_edges = 0;
        rrbag_[tid].sum_uncompressed_nodes = 0;
        rrbag_[tid].sum_uncompressed_edges = 0;

        rrbag_[tid].RR_heads.clear();
        rrbag_[tid].RR_nodes.clear();
        rrbag_[tid].first_parent.clear();
        rrbag_[tid].parents.clear();
        rrbag_[tid].first_child.clear();
        rrbag_[tid].children.clear();
        rrbag_[tid].RR_critical.clear();
        rrbag_[tid].nodes_boost_RR.resize(num_nodes_);
        rrbag_[tid].nodes_activate_RR.resize(num_nodes_);

        rrbag_[tid].score_init.resize(num_nodes_);
        std::fill(rrbag_[tid].score_init.begin(), rrbag_[tid].score_init.end(),
                  0);

        rrbag_[tid].bookmark_num_RR = 0;
        rrbag_[tid].bookmark_num_RR_boostable = 0;
    }
}

void RRSet::ResetGlobalStastics() {
    // Initialize global statistics
    num_RR_ = 0;
    num_RR_no_seed_ = 0;
    num_RR_activated_ = 0;
    num_RR_boostable_ = 0;
    num_nodes_to_boost_ = 0;
    num_RR_has_critical_ = 0;
    num_critical_nodes_ = 0;
    sum_nodes_ = 0;
    sum_edges_ = 0;
    sum_active_edges_ = 0;
    sum_uncompressed_nodes_ = 0;
    sum_uncompressed_edges_ = 0;
}

void RRSet::UpdateGlobalStatistics() {
    ResetGlobalStastics();
    // Initialize global statistics
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        num_RR_ += rrbag_[tid].num_RR;
        num_RR_no_seed_ += rrbag_[tid].num_RR_no_seed;
        num_RR_activated_ += rrbag_[tid].num_RR_activated;
        num_RR_boostable_ += rrbag_[tid].num_RR_boostable;
        num_nodes_to_boost_ += rrbag_[tid].num_nodes_to_boost;
        num_RR_has_critical_ += rrbag_[tid].num_RR_has_critical;
        num_critical_nodes_ += rrbag_[tid].num_critical_nodes;
        sum_nodes_ += rrbag_[tid].sum_nodes;
        sum_edges_ += rrbag_[tid].sum_edges;
        sum_active_edges_ += rrbag_[tid].sum_active_edges;
        sum_uncompressed_nodes_ += rrbag_[tid].sum_uncompressed_nodes;
        sum_uncompressed_edges_ += rrbag_[tid].sum_uncompressed_edges;
    }
}

// Read the graph and seeds, prepare RR sets
RRSet::RRSet(const char *graph_file, const char *seed_file, double beta, int k)
    : Boosting(graph_file, seed_file, beta, k) {
    InitRRBags();
    InitNewRRHelpers();
    ResetGlobalStastics();
}

RRSet::~RRSet() {}

#ifdef LBONLY
// LBONLY CompressRR
void RRSet::CompressRR(int root, int tid) {
    NewRRHelper &helper = genrr_[tid];
    RRBag &rrbag = rrbag_[tid];
    int RR_id = rrbag.num_RR_boostable;

    // Search for nodes reachable from seeds (distance == 0)
    // For every such node, check whether its parents are "critical nodes".
    // For other nodes, we don't care.
    // Initially, nodes in helper.dq2 are seeds
    rrbag.RR_critical.emplace_back(vector<int>());  // init
    while (!helper.dq2.empty()) {
        int u = helper.dq2.front().first;
        helper.dq2.pop_front();
        for (const auto &p : helper.parents[u]) {
            int p_node = p.node();
            if (helper.visited[p_node] != helper.visit_id) continue;
            if (p.distance() == 0) {
                // we can directly influence u's parents
                if (helper.dis_from_seeds[p_node] > 0) {
                    helper.dis_from_seeds[p_node] = 0;
                    helper.dq2.emplace_back(p_node, 0);
                }
            } else if (helper.dis_to_root[p_node] == 0) {
                /// u's parent is a critical node
                rrbag.score_init[p_node]++;
                rrbag.RR_critical.back().push_back(p_node);
                rrbag.nodes_activate_RR[p_node].push_back(RR_id);
                helper.dis_to_root[p_node] = kMaxDis;  // avoid duplication
            }
        }
    }

    // Update statistics
    if (rrbag.RR_critical.back().size() > 0) {
        rrbag.num_RR_has_critical++;
        rrbag.num_critical_nodes += rrbag.RR_critical.back().size();
    }
}
#else
// Normal mode: Compress RR
void RRSet::CompressRR(int root, int tid) {
    NewRRHelper &helper = genrr_[tid];
    RRBag &rrbag = rrbag_[tid];

    // Start the compression
    int first_node = rrbag.RR_nodes.size();
    rrbag.RR_heads.push_back(first_node);
    assert(rrbag.num_RR_boostable == (int)rrbag.RR_heads.size() - 1);
    int RR_id = rrbag.RR_heads.size() - 1;
    int last_node = first_node;

    // static int total_node = 0;
    // static int num_super_seed = 0;
    // static int num_rprime_valid = 0;

    // Compute distances from seed nodes, and reset distances to the root node
    while (!helper.dq2.empty()) {
        int u = helper.dq2.front().first;
        DistanceType d_u_from_seeds = helper.dq2.front().second;
        helper.dq2.pop_front();
        if (helper.dis_from_seeds[u] != d_u_from_seeds) continue;
        // total_node++;
        // if (helper.dis_from_seeds[u] == 0) {
        //     num_super_seed++;
        // }
        helper.dis_to_root[u] = kMaxDis;  // reset
        // Check parents
        for (const auto &p : helper.parents[u]) {
            int p_node = p.node();
            DistanceType d_p_from_seeds = d_u_from_seeds + p.distance();
            // No need to update the distance
            if (d_p_from_seeds > k_) continue;
            helper.children[p_node].emplace_back(u, p.type());
            if (d_p_from_seeds >= helper.dis_from_seeds[p_node]) continue;
            // Update Distance
            helper.dis_from_seeds[p_node] = d_p_from_seeds;
            assert(helper.dis_from_seeds[p_node] <= k_);
            // Update helper.dq2
            if (d_p_from_seeds > d_u_from_seeds) {
                helper.dq2.emplace_back(p_node, d_p_from_seeds);
            } else {
                helper.dq2.emplace_front(p_node, d_p_from_seeds);
            }
        }
        if (helper.dis_from_seeds[u] == 0) {
            // position for the seed node
            helper.node_pos[u] = first_node + 1;
        }
    }
    assert(!helper.children[root].empty());
    assert(helper.dis_from_seeds[root] != 0);
    assert(helper.dis_from_seeds[root] != kMaxDis);

    // Construct a new RR
    rrbag.RR_nodes.push_back(root);
    helper.node_pos[root] = last_node++;
    rrbag.first_parent.push_back(-1);
    rrbag.first_child.push_back(-1);

    rrbag.RR_nodes.push_back(kSuperSeed);
    last_node++;
    rrbag.first_parent.push_back(-1);
    rrbag.first_child.push_back(-1);

    // Compute distances to the root node
    // - Prepare node positions
    // - Update rrbag.RR_critical and score
    // - Count boostable nodes
    // - Add edges to the finalized RR set
    rrbag.RR_critical.emplace_back(vector<int>());
    helper.dis_to_root[root] = 0;
    helper.dq1.emplace_front(root, 0);
    while (!helper.dq1.empty()) {
        int u = helper.dq1.front().first;
        DistanceType d_u_to_root = helper.dq1.front().second;
        helper.dq1.pop_front();
        if (helper.dis_to_root[u] != d_u_to_root) continue;

        if (helper.dis_from_seeds[u] != 0 &&
            helper.dis_from_seeds[u] + helper.dis_to_root[u] <= k_) {
            if (helper.node_pos[u] < first_node) {
                rrbag.RR_nodes.push_back(u);
                helper.node_pos[u] = last_node++;
                rrbag.first_parent.push_back(-1);
                rrbag.first_child.push_back(-1);
                // num_rprime_valid++;
            }
            // Check children
            bool u_boost_and_done = false;
            bool u_boostable = false;
            // int u_pos = helper.node_pos[u];
            for (const auto &e : helper.children[u]) {
                int v = e.node();
                // Update nodes_boost_RR
                if (!u_boostable && !e.IsActive()) {
                    rrbag.nodes_boost_RR[u].push_back(RR_id);
                    u_boostable = true;
                    rrbag.num_nodes_to_boost++;
                }

                // Update score and critical
                if (helper.dis_from_seeds[v] == 0) {
                    if (!u_boost_and_done && d_u_to_root == 0) {
                        assert(e.IsActive() == false);
                        rrbag.score_init[u]++;
                        rrbag.RR_critical.back().push_back(u);
                        rrbag.nodes_activate_RR[u].push_back(RR_id);
                        u_boost_and_done = true;
                    }
                    continue;
                }

                // Update distance to root
                DistanceType d_v_to_root = d_u_to_root + e.distance();
                if (d_v_to_root > k_) continue;
                if (d_v_to_root >= helper.dis_to_root[v]) continue;
                helper.dis_to_root[v] = d_v_to_root;
                // Update helper.dq1
                if (d_v_to_root > d_u_to_root) {
                    helper.dq1.emplace_back(v, d_v_to_root);
                } else {
                    helper.dq1.emplace_front(v, d_v_to_root);
                }
            }  // for helper.children
        }
    }

    for (int u_pos = first_node; u_pos < last_node; u_pos++) {
        if (u_pos == first_node + 1) continue;  // seed node, no children
        int u = rrbag.RR_nodes[u_pos];
        if (u_pos == first_node) {
            assert(!helper.children[u].empty());
        }
        for (const auto &e : helper.children[u]) {
            int v = e.node();  // v is u's child
            int v_pos = helper.node_pos[v];
            if (v_pos < first_node) continue;
            rrbag.parents.emplace_back(u_pos, e.type(),
                                       rrbag.first_parent[v_pos]);
            rrbag.first_parent[v_pos] = rrbag.sum_edges;
            rrbag.children.emplace_back(v_pos, e.type(),
                                        rrbag.first_child[u_pos]);
            rrbag.first_child[u_pos] = rrbag.sum_edges;
            rrbag.sum_edges++;
            rrbag.sum_active_edges += e.IsActive();
        }
    }

    // Update statistics
    if (rrbag.RR_critical.back().size() > 0) {
        rrbag.num_RR_has_critical++;
        rrbag.num_critical_nodes += rrbag.RR_critical.back().size();
    }
}
#endif

// Generate a new RR set
// - Return 0 if it is boostable (store it in genrr_.*)
// - Return 1 if it is already activated
// - Return 2 if it has no seeds
int RRSet::NewRR(int tid) {
    NewRRHelper &helper = genrr_[tid];

    // Randomly select "root"
    int root = dsfmt_genrand_uint32(&helper.dsfmt_seed) % num_nodes_;
    if (is_seed_[root]) {
        return 1;  // already activated
    }

    // Initialize temporary RR set
    helper.visit_id++;  // assume helper.visit_id will always be smaller than
                        // INT_MAX
    helper.visited[root] = helper.visit_id;
    helper.dis_to_root[root] = 0;           // lazy init
    helper.dis_from_seeds[root] = kMaxDis;  // lazy init
    helper.parents[root].clear();
    helper.children[root].clear();

    // Double-ended queues
    while (!helper.dq1.empty()) helper.dq1.pop_front();
    while (!helper.dq2.empty()) helper.dq2.pop_front();
    helper.dq1.emplace_front(root, 0);

    double cnt_nodes = 1;
    double cnt_edges = 0;
    // Generate RR set (and compute helper.dis_to_root)
    // - Return 1 immediately if this RR set is activated
    // - Compute helper.parents. helperdis_to_root, ...
    // - Clear helper.children, helperdis_from_seeds, ...
    while (!helper.dq1.empty()) {
        int u = helper.dq1.front().first;
        DistanceType dis_u_to_root = helper.dq1.front().second;
        helper.dq1.pop_front();

        // Check whether the distance is outdated
        if (helper.dis_to_root[u] != dis_u_to_root) continue;

        // For each incoming edge of u, ...
        for (const auto &e : in_edges_[u]) {
            int v = e.node;  // v->u
            double r = dsfmt_genrand_open_close(&helper.dsfmt_seed);

            // Check whether if we should consider this edge
            if (r > e.pp) continue;

            RREdgeType edge_type = r < e.p ? kActive : kBoost;
            DistanceType dis_v_to_root = dis_u_to_root + (edge_type == kBoost);
#ifdef LBONLY
            if (dis_v_to_root > 1) continue;
#else
            if (dis_v_to_root > k_) continue;
#endif

            // Update distances
            bool distance_is_updated = false;
            if (helper.visited[v] != helper.visit_id) {
                // This is a new node
                helper.visited[v] = helper.visit_id;
                helper.dis_to_root[v] = dis_v_to_root;
                helper.parents[v].clear();
                helper.children[v].clear();
                distance_is_updated = true;
                if (is_seed_[v]) {
                    helper.dis_from_seeds[v] = 0;
                    helper.dq2.emplace_front(v, 0);
                } else {
                    helper.dis_from_seeds[v] = kMaxDis;
                }
                cnt_nodes++;
            } else if (dis_v_to_root < helper.dis_to_root[v]) {
                distance_is_updated = true;
                helper.dis_to_root[v] = dis_v_to_root;
            }

            if (is_seed_[v] && helper.dis_to_root[e.node] == 0) {
                return 1;  // This RR-set is already activated.
            }

            // Insert parent
            helper.parents[v].emplace_back(u, edge_type);
            cnt_edges++;

            // Update the queue
            if (distance_is_updated && !is_seed_[v]) {
                if (dis_v_to_root > dis_u_to_root) {
                    helper.dq1.emplace_back(v, dis_v_to_root);
                } else {
                    helper.dq1.emplace_front(v, dis_v_to_root);
                }
            }
        }  // for in_edges_
    }      // end of generating RR sets

    // This RR-set has no seed node
    if (helper.dq2.empty()) {
        return 2;
    }

    // only count uncompressed nodes/edges for boostable RR
    RRBag &rrbag = rrbag_[tid];
    rrbag.sum_uncompressed_nodes += cnt_nodes;
    rrbag.sum_uncompressed_edges += cnt_edges;

    CompressRR(root, tid);
    return 0;
}

// insert multiple RR sets
void RRSet::InsertRR(int num_new_RR) {
    if (num_new_RR <= 0) return;

    static double time_spent_here = 0.0;
    Timer tm_InsertRR;

// Insert RR sets
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        RRBag &rrbag = rrbag_[tid];
#pragma omp for schedule(static, 1)
        for (int count = 0; count < num_new_RR; count++) {
            // generate an RR set
            int ret = NewRR(tid);
            if (ret == 1) {
                rrbag.num_RR_activated++;
            } else if (ret == 2) {
                rrbag.num_RR_no_seed++;
            } else {
                rrbag.num_RR_boostable++;
            }
        }  // Done with generating RR sets
        rrbag.num_RR = rrbag.num_RR_activated + rrbag.num_RR_no_seed +
                       rrbag.num_RR_boostable;
        rrbag.sum_nodes = rrbag.RR_nodes.size();
    }

    time_spent_here += tm_InsertRR.TimeElapsed();
    UpdateGlobalStatistics();

    printf("[RRSet::InsertRR] Parallel Load (num_RR): ");
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        printf("%.2lf ", (double)rrbag_[tid].num_RR / num_RR_);
    }
    printf("\n");

    printf("[RRSet::InsertRR] num_RR_ = %d, act=%.2lf nos=%.2lf boo=%.2lf\n",
           num_RR_, (double)num_RR_activated_ / num_RR_,
           (double)num_RR_no_seed_ / num_RR_,
           (double)num_RR_boostable_ / num_RR_);
    printf("[RRSet::InsertRR] cumulative time = %.2lf (s)\n", time_spent_here);
    Timer::PrintCurrentTime("[RRSet::InsertRR] finished at ");
}

void RRSet::InsertRRBookmark() {
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        rrbag_[tid].bookmark_num_RR = rrbag_[tid].num_RR;
        rrbag_[tid].bookmark_num_RR_boostable = rrbag_[tid].num_RR_boostable;
    }
}

#ifndef LBONLY
// compute the boosted influence of the given boost set, using generated RR sets
double RRSet::ComputeBoost(const vector<int> &boost,
                           bool after_bookmark) const {
    // Flags for boosted rrbag.RR_nodes
    vector<bool> is_boost(num_nodes_, false);
    for (auto boost_node : boost) {
        if (boost_node >= 0) {
            is_boost[boost_node] = true;
        }
    }

    // queue
    double total_covered = 0;
    int first_RR = 0;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const RRBag &rrbag = rrbag_[tid];

        // All RR sets that contain boosted rrbag.RR_nodes
        set<int> related_RR_sets;
        for (auto boost_node : boost) {
            if (boost_node < 0) continue;
            related_RR_sets.insert(rrbag.nodes_boost_RR[boost_node].begin(),
                                   rrbag.nodes_boost_RR[boost_node].end());
        }

        int first_RR_boostable =
            after_bookmark ? rrbag.bookmark_num_RR_boostable : 0;
        double covered = 0;
        vector<int> in_queue(num_nodes_, -1);

        // iterate over all RR sets, see whether the root could be activated
        for (int RR_idx : related_RR_sets) {
            if (RR_idx < first_RR_boostable) continue;
            // starting from rrbag.RR_nodes reachable from seeds, to see whether
            // we could activate this RR set
            bool success = false;
            int first_node = rrbag.RR_heads[RR_idx];
            queue<int> q;
            q.push(first_node + 1);  // seed_node

            while (!q.empty() && !success) {
                int u = q.front();
                q.pop();
                for (int eid = rrbag.first_parent[u]; eid != -1;
                     eid = rrbag.parents[eid].get_next()) {
                    const auto &p = rrbag.parents[eid];
                    int v = p.node();
                    assert(rrbag.RR_nodes[v] != kSuperSeed);
                    if (in_queue[rrbag.RR_nodes[v]] == RR_idx) continue;
                    if (p.IsActive() || is_boost[rrbag.RR_nodes[v]]) {
                        if (v == first_node) {
                            success = true;
                            break;
                        }
                        q.push(v);
                        in_queue[rrbag.RR_nodes[v]] = RR_idx;
                    }
                }
            }
            if (success) {
                covered++;
            }
        }

#pragma omp atomic
        total_covered += covered;
#pragma omp atomic
        first_RR += after_bookmark ? rrbag.bookmark_num_RR : 0;
    }

    return (double)num_nodes_ * total_covered / (num_RR_ - first_RR);
}

void RRSet::CheckCriticalNodes(const vector<int> &boost) const {
    // Flags for boosted rrbag.RR_nodes
    vector<bool> is_boost(num_nodes_, false);
    for (auto boost_node : boost) {
        if (boost_node >= 0) {
            is_boost[boost_node] = true;
        }
    }

    double total_covered = 0;
    // number of activated boostable RR sets with critical nodes
    double total_covered_have_critical = 0;
    // number of activated boostable RR sets with critical nodes boosted
    double total_covered_have_critical_boosted = 0;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const RRBag &rrbag = rrbag_[tid];

        // All RR sets that contain boosted rrbag.RR_nodes
        set<int> related_RR_sets;
        for (auto boost_node : boost) {
            if (boost_node < 0) continue;
            related_RR_sets.insert(rrbag.nodes_boost_RR[boost_node].begin(),
                                   rrbag.nodes_boost_RR[boost_node].end());
        }
        double covered = 0;
        double covered_have_critical = 0;
        double covered_have_critical_boosted = 0;

        vector<int> in_queue(num_nodes_, -1);
        // iterate over all RR sets, see whether the root could be activated
        for (int RR_idx : related_RR_sets) {
            // starting from rrbag.RR_nodes reachable from seeds, to see whether
            // we could activate this RR set
            bool success = false;
            int first_node = rrbag.RR_heads[RR_idx];
            queue<int> q;
            q.push(first_node + 1);  // seed_node

            while (!q.empty() && !success) {
                int u = q.front();
                q.pop();
                for (int eid = rrbag.first_parent[u]; eid != -1;
                     eid = rrbag.parents[eid].get_next()) {
                    const auto &p = rrbag.parents[eid];
                    int v = p.node();
                    assert(rrbag.RR_nodes[v] != kSuperSeed);
                    if (in_queue[rrbag.RR_nodes[v]] == RR_idx) continue;
                    if (p.IsActive() || is_boost[rrbag.RR_nodes[v]]) {
                        if (v == first_node) {
                            success = true;
                            break;
                        }
                        q.push(v);
                        in_queue[rrbag.RR_nodes[v]] = RR_idx;
                    }
                }
            }
            if (success) {
                covered++;
                if (!rrbag.RR_critical[RR_idx].empty()) {
                    covered_have_critical++;
                }
            }
        }

        for (int RR_idx : related_RR_sets) {
            if (!rrbag.RR_critical[RR_idx].empty()) {
                for (auto u : rrbag.RR_critical[RR_idx]) {
                    if (is_boost[u]) {
                        covered_have_critical_boosted++;
                        break;
                    }
                }
            }
        }

#pragma omp atomic
        total_covered += covered;

#pragma omp atomic
        total_covered_have_critical += covered_have_critical;

#pragma omp atomic
        total_covered_have_critical_boosted += covered_have_critical_boosted;
    }

    printf("[RRSet::CheckCriticalNodes] covered boostable PRR-graphs: with critical nodes/all = %.2lf\n",
           total_covered_have_critical / total_covered);
    printf("[RRSet::CheckCriticalNodes] covered boostable PRR-graphs with critical nodes: boosted / all = %.2lf\n",
           total_covered_have_critical_boosted/total_covered_have_critical);
    printf("[RRSet::CheckCriticalNodes] mu(B) = %.2lf\n",
           (double)num_nodes_ * total_covered_have_critical_boosted / num_RR_);
    printf("[RRSet::CheckCriticalNodes] Delta(B) = %.2lf\n",
           (double)num_nodes_ * total_covered / num_RR_);
    printf("[RRSet::CheckCriticalNodes] mu(B) / Delta(B) = %.2lf\n",
           total_covered_have_critical_boosted / total_covered);
}
#endif

// Single threaded... (useless?)
// Generate another set of RR sets to evaluation the true boost
double RRSet::ReComputeBoost(const vector<int> &boost, int num_RR) {
    // Flags for boosted rrbag.RR_nodes
    vector<bool> is_boost(num_nodes_, false);
    for (auto boost_node : boost) {
        if (boost_node >= 0) {
            is_boost[boost_node] = true;
        }
    }

    double covered = 0;
    int num_activated = 0;

    vector<DistanceType> dis_to_root(num_nodes_, kMaxDis);
    queue<int> q1;
    queue<int> q2;

    dsfmt_t dsfmt_seed;
    dsfmt_init_gen_rand(&dsfmt_seed, time(NULL));

    // iterate over all rr sets, see whether the root could be activated
    for (int rr_idx = 0; rr_idx < num_RR; rr_idx++) {
        // get ready
        int root = dsfmt_genrand_uint32(&dsfmt_seed) % num_nodes_;
        if (is_seed_[root]) {
            num_activated++;
            continue;  // originally activated
        }

        std::fill(dis_to_root.begin(), dis_to_root.end(), kMaxDis);  // unknown
        while (!q1.empty()) q1.pop();
        while (!q2.empty()) q2.pop();
        q1.push(root);
        dis_to_root[root] = 0;

        bool activated = false;
        bool success_boost = false;

        // generate rr set (expand rrbag.RR_nodes that can activate
        // root)
        while (!activated && !q1.empty()) {
            int u = q1.front();
            q1.pop();
            // for each incoming edge of e, ...
            for (const auto &e : in_edges_[u]) {
                double r = dsfmt_genrand_open_close(&dsfmt_seed);
                if (r > e.pp) continue;
                // if we should consider this edge
                RREdgeType edge_type = r < e.p ? kActive : kBoost;
                if (edge_type == kActive) {
                    if (is_seed_[e.node]) {
                        activated = true;
                        break;  // break for
                    }
                    if (dis_to_root[e.node] != 0) {
                        dis_to_root[e.node] = 0;
                        q1.push(e.node);
                    }
                } else if (!success_boost) {
                    // edge_type == kBoost
                    if (is_boost[u]) {
                        if (is_seed_[e.node]) {
                            success_boost = true;
                        }
                        if (dis_to_root[e.node] > 1) {
                            dis_to_root[e.node] = 1;
                            q2.push(e.node);
                        }
                    }
                }
            }
        }

        if (activated) {
            num_activated++;
            continue;
        }

        // generate rr set (expand rrbag.RR_nodes that can activate root
        // with boost)
        while (!success_boost && !q2.empty()) {
            int u = q2.front();
            q2.pop();
            if (dis_to_root[u] != 1) continue;
            // for each incoming edge of e, ...
            for (const auto &e : in_edges_[u]) {
                double r = dsfmt_genrand_open_close(&dsfmt_seed);
                if (r > e.pp) continue;
                if (r > e.p && !is_boost[u]) {
                    continue;
                }
                // this edges is active (without boost or after boost)
                if (is_seed_[e.node]) {
                    success_boost = true;
                    break;  // break for
                }
                if (dis_to_root[e.node] > 1) {
                    dis_to_root[e.node] = 1;
                    q2.push(e.node);
                }
            }
        }

        if (success_boost) {
            covered++;
            continue;
        }
    }  // finished iterating all rr sets

    printf("num_activated = %d (%.2lf%%, inf = %.2lf)\n", num_activated,
           100.0 * num_activated / num_RR,
           (double)num_nodes_ * num_activated / num_RR);

    return (double)num_nodes_ * covered / num_RR;
}

#ifndef LBONLY
// Return the boosted influence spread
// If "before_bookmark" is ture,
// RR-sets generated after(>=) "the bookmark" will be ignored.
void RRSet::NodeSelectionTrue(vector<int> &boost, double &boost_true_score,
                              bool before_bookmark) const {
    double sum_score = 0;
    boost.resize(k_);
    std::fill(boost.begin(), boost.end(), -1);
    vector<bool> is_boost(num_nodes_, false);
    vector<double> score[NUM_THREADS];

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const RRBag &rrbag = rrbag_[tid];

        // Get ready
        int last_RR_boostable = before_bookmark
                                    ? rrbag.bookmark_num_RR_boostable
                                    : rrbag.num_RR_boostable;
        queue<int> q;

        vector<bool> rr_covered(rrbag.RR_heads.size(), false);

        vector<bool> tmp_can_activate_root(rrbag.RR_nodes.size(), false);
        vector<bool> tmp_activated_by_seeds(rrbag.RR_nodes.size(), false);

        for (int rr_idx = 0; rr_idx < last_RR_boostable; rr_idx++) {
            int first_node = rrbag.RR_heads[rr_idx];
            q.push(first_node);
            tmp_can_activate_root[first_node] = true;
            tmp_activated_by_seeds[first_node + 1] = true;
        }
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int eid = rrbag.first_child[u]; eid != -1;
                 eid = rrbag.children[eid].get_next()) {
                const auto &c = rrbag.children[eid];
                int v = c.node();
                if (c.type() == kActive && !tmp_can_activate_root[v]) {
                    tmp_can_activate_root[v] = true;
                    q.push(v);
                }
            }
        }

        // Compute initial score (this thread)
        score[tid].resize(num_nodes_, 0.0);
        for (int rr_idx = 0; rr_idx < last_RR_boostable; rr_idx++) {
            for (int u : rrbag.RR_critical[rr_idx]) {
                score[tid][u]++;
            }
        }

        for (int i = 0; i < k_; i++) {
// Wait.
#pragma omp barrier
#pragma omp master
            {
                // Select the best node
                double best_sum_score = -1;
                for (int u = 0; u < num_nodes_; u++) {
                    double tmp_score = 0.0;
                    for (int t = 0; t < NUM_THREADS; t++) {
                        tmp_score += score[t][u];
                    }
                    if (tmp_score > best_sum_score) {
                        best_sum_score = tmp_score;
                        boost[i] = u;
                    }
                }
                sum_score += best_sum_score;
                is_boost[boost[i]] = true;
            }
// Wait.
#pragma omp barrier

            score[tid][boost[i]] = -1;

            // Update score (stupid implementation)
            if (i == k_ - 1) continue;
            for (int rr_idx : rrbag.nodes_boost_RR[boost[i]]) {
                if (rr_idx >= last_RR_boostable || rr_covered[rr_idx]) continue;

                int first_node = rrbag.RR_heads[rr_idx];
                int last_node = rr_idx < (int)rrbag.RR_heads.size() - 1
                                    ? rrbag.RR_heads[rr_idx + 1]
                                    : rrbag.RR_nodes.size();
                int boost_pos = -1;

                assert(!tmp_activated_by_seeds[first_node]);
                assert(!tmp_can_activate_root[first_node + 1]);

                // deduct previous score
                for (int u = first_node; u < last_node; u++) {
                    if (rrbag.RR_nodes[u] == boost[i]) boost_pos = u;
                    if (!tmp_can_activate_root[u]) continue;
                    assert(!tmp_activated_by_seeds[u]);
                    for (int eid = rrbag.first_child[u]; eid != -1;
                         eid = rrbag.children[eid].get_next()) {
                        const auto &c = rrbag.children[eid];
                        int v = c.node();
                        if (tmp_activated_by_seeds[v]) {
                            // seed --a--> v --b--> u --a--> root
                            assert(c.type() != kActive);
                            assert(!is_boost[rrbag.RR_nodes[u]] ||
                                   rrbag.RR_nodes[u] == boost[i]);
                            // if we boost j, we can cover this rr set
                            score[tid][rrbag.RR_nodes[u]]--;
                            if (rrbag.RR_nodes[u] == boost[i]) {
                                rr_covered[rr_idx] = true;
                            }
                            break;  // we have finished the udpate of u
                        }
                    }
                }  // finish deduct previous score
                assert(boost_pos != first_node + 1);

                if (rr_covered[rr_idx]) {
                    continue;  // check the next rr set
                }

                // update "tmp_can_activate_root"
                assert(q.empty());
                if (tmp_can_activate_root[boost_pos]) {
                    q.push(boost_pos);
                }
                while (!q.empty()) {
                    int u = q.front();
                    q.pop();
                    assert(rrbag.RR_nodes[u] != kSuperSeed);
                    for (int eid = rrbag.first_child[u]; eid != -1;
                         eid = rrbag.children[eid].get_next()) {
                        const auto &c = rrbag.children[eid];
                        int v = c.node();
                        if (tmp_can_activate_root[v]) continue;
                        if (c.IsActive() || is_boost[rrbag.RR_nodes[u]]) {
                            tmp_can_activate_root[v] = true;
                            q.push(v);
                        }
                    }
                }

                // update "tmp_activated_by_seeds"
                assert(q.empty());
                if (!tmp_activated_by_seeds[boost_pos]) {
                    for (int eid = rrbag.first_child[boost_pos]; eid != -1;
                         eid = rrbag.children[eid].get_next()) {
                        const auto &c = rrbag.children[eid];
                        int v = c.node();
                        if (tmp_activated_by_seeds[v]) {
                            tmp_activated_by_seeds[boost_pos] = true;
                            q.push(boost_pos);
                            break;
                        }
                    }
                }
                while (!q.empty()) {
                    int u = q.front();
                    q.pop();
                    for (int eid = rrbag.first_parent[u]; eid != -1;
                         eid = rrbag.parents[eid].get_next()) {
                        const auto &p = rrbag.parents[eid];
                        int v = p.node();
                        if (tmp_activated_by_seeds[v]) continue;
                        // rrbag.RR_nodes[v] != kSuperSeed
                        if (p.IsActive() || is_boost[rrbag.RR_nodes[v]]) {
                            assert(!tmp_can_activate_root[v]);
                            tmp_activated_by_seeds[v] = true;
                            q.push(v);
                        }
                    }
                }

                assert(!tmp_activated_by_seeds[first_node]);
                assert(!tmp_can_activate_root[first_node + 1]);

                // add back score
                for (int u = first_node; u < last_node; u++) {
                    if (u == first_node + 1) continue;
                    if (!tmp_can_activate_root[u]) continue;
                    if (is_boost[rrbag.RR_nodes[u]]) continue;
                    for (int eid = rrbag.first_child[u]; eid != -1;
                         eid = rrbag.children[eid].get_next()) {
                        const auto &c = rrbag.children[eid];
                        int v = c.node();
                        if (!tmp_activated_by_seeds[v]) continue;
                        // [seed] --a--> v --b--> u --a--> root
                        assert(c.type() == kBoost);
                        assert(!is_boost[rrbag.RR_nodes[u]]);
                        // if we boost j, we can cover this rr set
                        score[tid][rrbag.RR_nodes[u]]++;
                        break;  // we have finished the udpate for u
                    }
                }
            }  // finished update scores
        }
    }

    int last_RR = 0;
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        last_RR +=
            before_bookmark ? rrbag_[tid].bookmark_num_RR : rrbag_[tid].num_RR;
    }

    boost_true_score = num_nodes_ * sum_score / last_RR;
}
#endif

// Using a single thread (fast already)
// Select top-k boost rrbag.RR_nodes according to score_lb
// Return the boosted influence spread
void RRSet::NodeSelectionLowerBound(vector<int> &boost,
                                    double &boost_lb_lb_score) const {
    printf("[RRSet::NodeSelectionLowerBound] Start...\n");
    // Initialize score
    vector<double> score(num_nodes_, 0.0);
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        for (int u = 0; u < num_nodes_; u++) {
            score[u] += rrbag_[tid].score_init[u];
        }
    }
    boost.resize(k_);
    std::fill(boost.begin(), boost.end(), -1);

    double sum_score_lb = 0.0;
    vector<bool> rr_covered[NUM_THREADS];
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        rr_covered[tid].resize(num_RR_boostable_, false);
    }

    for (int i = 0; i < k_; i++) {
        // select the best node
        auto id = max_element(score.begin(), score.end());
        sum_score_lb += *id;
        boost[i] = id - score.begin();
        *id = -1;

        if (i == k_ - 1) continue;

        // update score
        for (int tid = 0; tid < NUM_THREADS; tid++) {
            for (auto rr_idx : rrbag_[tid].nodes_activate_RR[boost[i]]) {
                if (rr_covered[tid][rr_idx]) {
                    continue;
                }
                for (int u : rrbag_[tid].RR_critical[rr_idx]) {
                    if (boost[i] == u) {
                        rr_covered[tid][rr_idx] = true;
                        break;
                    }
                }
                if (rr_covered[tid][rr_idx]) {
                    for (int u : rrbag_[tid].RR_critical[rr_idx]) {
                        score[u]--;
                    }
                }
            }
        }  // finish updating score
    }
    boost_lb_lb_score = num_nodes_ * sum_score_lb / num_RR_;
    printf("[RRSet::NodeSelectionLowerBound] lower bound of boost = %.2lf\n",
           boost_lb_lb_score);
}

void RRSet::PrintStatistics() const {
    printf("num_RR_ = %d\n", num_RR_);

    printf("num_RR_no_seed_ = %d (%.2lf%%)\n", num_RR_no_seed_,
           100.0 * num_RR_no_seed_ / num_RR_);

    printf("num_RR_activated_= %d (%.2lf%%)\n", num_RR_activated_,
           100.0 * num_RR_activated_ / num_RR_);

    printf("num_RR_boostable_ = %d (%.2lf%%)\n", num_RR_boostable_,
           100.0 * num_RR_boostable_ / num_RR_);

    printf("num_boostable_rr_has_critical = %d (%.2lf%%)\n",
           num_RR_has_critical_,
           100.0 * num_RR_has_critical_ / num_RR_boostable_);

    printf("sum_nodes_ = %.0lf, sum_edges_ = %.0lf\n", sum_nodes_, sum_edges_);
    printf("uncompressed: sum_nodes_ = %.0lf, sum_edges_ = %.0lf\n",
           sum_uncompressed_nodes_, sum_uncompressed_edges_);
    printf("ave_nodes = %.2lf, ave_edges = %.2lf\n",
           sum_nodes_ / num_RR_boostable_, sum_edges_ / num_RR_boostable_);
    printf("uncompressed: ave_nodes = %.2lf, ave_edges = %.2lf\n",
           sum_uncompressed_nodes_ / num_RR_boostable_,
           sum_uncompressed_edges_ / num_RR_boostable_);
    printf("ave_active_edges = %.2lf\n", sum_active_edges_ / num_RR_boostable_);
    printf("ave_nodes_to_boost = %.2lf\n",
           num_nodes_to_boost_ / num_RR_boostable_);
    printf("ave_critical = %.2lf\n",
           num_critical_nodes_ / num_RR_has_critical_);
}

double RRSet::ComputeBoostLowerBound(const vector<bool> &is_boost,
                                     bool after_bookmark) const {
    double cover = 0;
    int first_RR = 0;
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        int first_RR_boostable =
            after_bookmark ? rrbag_[tid].bookmark_num_RR_boostable : 0;
        first_RR += after_bookmark ? rrbag_[tid].bookmark_num_RR : 0;
        for (int i = first_RR_boostable; i < rrbag_[tid].num_RR_boostable;
             i++) {
            for (const auto &u : rrbag_[tid].RR_critical[i]) {
                if (is_boost[u]) {
                    cover++;
                    break;
                }
            }
        }
    }
    return num_nodes_ * cover / (num_RR_ - first_RR);
}

double RRSet::ComputeBoostLowerBound(const vector<int> &boost,
                                     bool after_bookmark) const {
    // Flags for boosted rrbag.RR_nodes
    static vector<bool> is_boost(num_nodes_, false);
    for (auto boost_node : boost) {
        if (boost_node >= 0) {
            is_boost[boost_node] = true;
        }
    }
    double result = ComputeBoostLowerBound(is_boost, after_bookmark);
    for (auto boost_node : boost) {
        if (boost_node >= 0) {
            is_boost[boost_node] = false;
        }
    }
    return result;
}

#ifndef LBONLY
void RRSet::LowerBoundTest(int test_size, const char *lb_log_filename,
                           const char *graph_file, double epsilon) {
    printf("[RRSet::LowerBoundTest] Start...\n");

    vector<int> boost(k_, -1);
    vector<double> boost_true(test_size, 0);
    vector<double> boost_lb(test_size, 0);
    vector<double> ratio(test_size, 0);
    double ratio_min = 1.0;
    double ratio_max = 0.0;

    // Set with larger boost
    NodeSelectionLowerBound(boost, boost_lb[0]);
    boost_true[0] = ComputeBoost(boost);
    ratio[0] = (boost_true[0] < kEPS) ? 1.0 : (boost_lb[0] / boost_true[0]);

    // Initialize the random seed
    dsfmt_t dsfmt_seed;
    dsfmt_init_gen_rand(&dsfmt_seed, time(NULL));

    // Randomly remove a node and add another one
    for (int i = 1; i < test_size; i++) {
        // Generate the number of replaced position
        int num_new = dsfmt_genrand_uint32(&dsfmt_seed) % k_;
        vector<int> boost_new(boost);
        while (num_new > 0) {
            // Generate a position
            int pos = dsfmt_genrand_uint32(&dsfmt_seed) % k_;
            // Check whether the position has been changed
            if (boost[pos] != boost_new[pos]) continue;
            boost_new[pos] = -1;
            while (boost_new[pos] == -1) {
                // Generate a new node id
                int u = dsfmt_genrand_uint32(&dsfmt_seed) % num_nodes_;
                if (is_seed_[u]) continue;
                boost_new[pos] = u;
                for (int j = 0; j < k_ && boost_new[pos] != -1; j++) {
                    if (j != pos && boost_new[pos] == boost_new[j]) {
                        boost_new[pos] = -1;
                    }
                }
                // If boost_new[pos] != -1, done.
            }
            num_new--;
        }
        // Test the tightness of lower bound
        boost_true[i] = ComputeBoost(boost_new);
        boost_lb[i] = ComputeBoostLowerBound(boost_new);
        ratio[i] = (boost_true[i] < kEPS) ? 1.0 : (boost_lb[i] / boost_true[i]);
        // Output
        ratio_min = std::min(ratio_min, ratio[i]);
        ratio_max = std::max(ratio_max, ratio[i]);
    }

    // Sort the detailed ratio
    vector<int> idx(test_size, 0);
    for (int i = 0; i < test_size; i++) {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return boost_true[a] < boost_true[b]; });

    // Print the detailed ratio
    char temp_filename[128];
    sprintf(temp_filename, "lb_log_tmp/lb_log_n%d_b%.0lf_k%d_eps%.0lf_%ld.txt",
            num_nodes_, beta_, k_, 100.0 * epsilon, time(0));
    FILE *f_ratio = fopen(temp_filename, "w");
    if (f_ratio == NULL) {
        printf("Cannot open lb_log file %s!\n", temp_filename);
        exit(-1);
    }
    for (int i = 0; i < test_size; i++) {
        fprintf(f_ratio, "%.6lf %.6lf %.6lf\n", boost_lb[idx[i]],
                boost_true[idx[i]], ratio[idx[i]]);
    }
    fclose(f_ratio);

    double ratio_mean =
        std::accumulate(ratio.begin(), ratio.end(), 0.0) / ratio.size();
    double ratio_sq_sum =
        std::inner_product(ratio.begin(), ratio.end(), ratio.begin(), 0.0);
    double ratio_stdev =
        std::sqrt(ratio_sq_sum / ratio.size() - ratio_mean * ratio_mean);

    double boost_true_mean =
        std::accumulate(boost_true.begin(), boost_true.end(), 0.0) /
        boost_true.size();
    double boost_true_sq_sum = std::inner_product(
        boost_true.begin(), boost_true.end(), boost_true.begin(), 0.0);
    double boost_true_stdev = std::sqrt(boost_true_sq_sum / boost_true.size() -
                                        boost_true_mean * boost_true_mean);

    double corr = 0.0;
    for (unsigned int i = 0; i < boost_true.size(); i++) {
        corr += (boost_true[i] - boost_true_mean) * (ratio[i] - ratio_mean);
    }
    corr /= num_nodes_;
    corr /= (ratio_stdev * boost_true_stdev);

    // print summary
    FILE *f_lb_log = fopen(lb_log_filename, "a");
    if (f_lb_log == NULL) {
        printf("Cannot open %s\n", lb_log_filename);
        exit(-1);
    }
    fprintf(f_lb_log, "%s\t", graph_file);
    fprintf(f_lb_log, "%d\t%d\t%.2lf\t", num_nodes_, num_edges_,
            (double)num_edges_ / num_nodes_);
    fprintf(f_lb_log, "%d\t%.0lf\t%d\t%d\t", num_seeds_, beta_, k_, num_RR_);
    fprintf(f_lb_log, "%s\t", temp_filename);
    fprintf(f_lb_log, "%.2lf\t%.2lf\t%.2lf\t%.2lf\t", ratio_mean, ratio_min,
            ratio_max, ratio_stdev);
    fprintf(f_lb_log, "%.2lf\n", corr);

    fclose(f_lb_log);
    printf("[RRSet::LowerBoundTest] Done...\n");
}

#endif
