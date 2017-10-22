#!/bin/bash

graph="digg_learn"
seed=50
beta=2
k=100
epsilon=1

# PRR-Boost 
mkdir -p lb_log_tmp # the folder name is harded-coded...
./bin/prrboost \
    dataset/${graph}/graph_ic_nm.inf \
    dataset/${graph}/seeds${seed}.txt \
    ${beta} \
    ${k} \
    ${epsilon} \
    log_prrboost.txt \
    log_prrboost_lbtest.txt

# PRR-Boost-LB
./bin/prrboost_lb \
    dataset/${graph}/graph_ic_nm.inf \
    dataset/${graph}/seeds${seed}.txt \
    ${beta} \
    ${k} \
    ${epsilon} \
    log_prrboost_lb.txt

# Heuristic methods (except for "MoreSeeds")
./bin/heu \
    dataset/${graph}/graph_ic_nm.inf \
    dataset/${graph}/seeds${seed}.txt \
    ${beta} \
    ${k} \
    log_heu.txt 

# Heuristic methods: MoreSeeds
./bin/moreseeds \
    dataset/${graph}/graph_ic_nm.inf \
    dataset/${graph}/seeds${seed}.txt \
    ${beta} \
    ${k} \
    ${epsilon} \
    log_moreseeds.txt 

