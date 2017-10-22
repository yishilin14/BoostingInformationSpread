#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "dsfmt/dSFMT.h"
#include "getRSS.h"
#include "heuristics.h"
#include "timer.h"

using std::set;
using std::string;
using std::vector;
using std::queue;

// Parameters: graph_nm seed_filename k epsilon log_file
int main(int argc, const char **argv) {
    const char *graph_filename;
    const char *seed_filename;
    const char *log_filename;

    if (argc < 5) {
        printf(
            "Usage: %s graph_filename, seed_filename, beta, k, "
            "log_filename(optional)\n",
            argv[0]);
        exit(-1);
    }
    graph_filename = argv[1];
    seed_filename = argv[2];
    int beta = atoi(argv[3]);
    int k = atoi(argv[4]);
    log_filename = argc > 5 ? argv[5] : NULL;

    printf("[Parameters] %s %s beta=%d, k=%d\n", graph_filename, seed_filename,
           beta, k);

    Heuristics heu = Heuristics(graph_filename, seed_filename, beta, k);

    vector<int> boost(k);
    vector<double> boost_inf;

    Timer::PrintCurrentTime("[Start time]");

    int num_MC = 20000;  // default
    // int num_MC = 100;  // debug

#ifndef INFONLY
    Timer timer;
    double out, discount;
    // Out weighted degree global (e.p)
    timer.Reset();
    out = true, discount = false;
    heu.HighDegreeGlobal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree global (out)]");

    // Out weighted degree local (e.p)
    timer.Reset();
    out = true, discount = false;
    heu.HighDegreeLocal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree local (out)]");

    // Out discounted weighted degree global (e.p)
    timer.Reset();
    out = true, discount = true;
    heu.HighDegreeGlobal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree global (out, discount)]");

    // Out discounted weighted degree local (e.p)
    timer.Reset();
    out = true, discount = true;
    heu.HighDegreeLocal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree local (out, discount)]");

    // In weighted degree global (e.p)
    timer.Reset();
    out = false, discount = false;
    heu.HighDegreeGlobal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree global (in)]");

    // In weighted degree local (e.p)
    timer.Reset();
    out = false, discount = false;
    heu.HighDegreeLocal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree local (in)]");

    // In discounted weighted degree global (e.p)
    timer.Reset();
    out = false, discount = true;
    heu.HighDegreeGlobal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree global (out, discount)]");

    // In discounted weighted degree local (e.p)
    timer.Reset();
    out = false, discount = true;
    heu.HighDegreeLocal(boost, out, discount);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[High degree local (out, discount)]");

    // PageRank
    timer.Reset();
    heu.PageRank(boost);
    timer.PrintTimeElapsed();
    boost_inf.push_back(heu.ComputeBoostMC(boost, num_MC));
    Timer::PrintCurrentTime("[PageRank]");
#endif

    // Influence 
    boost_inf.push_back(heu.ComputeSeedInf(num_MC));
    Timer::PrintCurrentTime("[Seed influence]");

    heu.PrintLogToFile(log_filename, graph_filename, boost_inf);

    Timer::PrintCurrentTime("[End time]");
    return 0;
}
