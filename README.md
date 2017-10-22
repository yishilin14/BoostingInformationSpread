# Boosting Information Spread 

Implementations of *Boosting information spread: An algorithmic approach*.  For more details about PRR-Boost and PRR-Boost, please refer to our paper:

Yishi Lin, Wei Chen, John C.S. Lui. Boosting information spread: An algorithmic approach. (ICDE'17)

## Compile
Use the `make' command to compile everything.

## How to use

- Please refer to `run.sh` for examples (e.g., parameters, output log files).

### Input format of graphs

- An example: 'datasets/digg_learn/graph_ic_nm.inf'
- The first line contains the number of nodes `n` and the number of edges `m`. Each of the
  remaining `m` lines contains three values `u`, `v` and `p` meaning that the
  influence probability from `u` to `v` is `p` (Independent Cascade model).

### Input format of seeds 

- An example: 'datasets/digg_learn/seeds50.txt'
- Each line contains a seed node.

### Output format of log files

#### PRR-Boost

The first log file of `./bin/prrboost` contains the following information.
- Columns 1-7: dataset, # nodes, # edges, # seeds, beta, k, epsilon
- Columns 13-15: the average number of edges, uncompressed nodes and uncompressed edges of boostable PRR-graphs
- Columns 24-27: running time, memory usage, boost of influence spread, initial memory
  usage before generating PRR-graphs

The second log file of `./bin/prrboost` contains information about parameters (first seven columns) and the name of a file containing data used to plot Figures such as Figure 6 and Figure 8 in the paper (lower bound, boosted influence, and their ratios).

#### PRR-Boost-LB

Log file of `./bin/prrboost_lb`:
- Columns 1-7: dataset, # nodes, # edges, # seeds, beta, k, epsilon
- Columns 24-27: running time, memory usage, boost of influence spread, initial memory
  usage before generating PRR-graphs

#### Heirustic methods

Log file of `./bin/heu`:
- Columns 1-4: dataset, # seeds, beta, k
- Columns 5-13: HighDegreeGlobalOut, HighDegreeLocalOut, HighDegreeGlobalOutDiscount, HighDegreeLocalOutDiscount, HighDegreeGlobalIn, HighDegreeLocalIn, HighDegreeGlobalInDiscount, HighDegreeLocalInDiscount, PageRank

Log file of `./bin/moreseeds`:
- Columns 1-7: dataset, # nodes, # edges, # seeds, beta, k, epsilon
- Columns 11-13: boost of influence spread, running time, memory usage
