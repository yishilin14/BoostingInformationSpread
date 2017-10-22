CXX = g++-7
CPPFLAGS = -std=c++11 -O3 -Wall -DDSFMT_MEXP=19937 -msse2 -DHAVE_SSE2=1 -fopenmp -DNUM_THREADS=8 -DNDEBUG

all: prrboost prrboost_lb heu moreseeds

# PRR-Boost
prrboost: src/main_prrboost.cpp src/boosting.* src/rrset.* 
	${CXX} ${CPPFLAGS} -DTESTLB -DRECOMPUTE src/main_prrboost.cpp \
		src/rrset.cpp src/boosting.cpp src/getRSS.cpp \
		src/dsfmt/dSFMT.c -o bin/prrboost

# PRR-Boost-LB
prrboost_lb: src/main_prrboost.cpp src/boosting.* src/rrset.* 
	${CXX} ${CPPFLAGS} -DLBONLY -DRECOMPUTE src/main_prrboost.cpp \
		src/rrset.cpp src/boosting.cpp src/getRSS.cpp \
		src/dsfmt/dSFMT.c -o bin/prrboost_lb

# Heuristic methods (degree-based)
heu: src/main_heuristics.cpp src/boosting.* 
	${CXX} ${CPPFLAGS} src/main_heuristics.cpp src/heuristics.h \
		src/boosting.cpp src/getRSS.cpp src/dsfmt/dSFMT.c -o bin/heu

# Heuristic method: use the IM method to select more "seeds", and let them be boosted users
moreseeds: src/main_moreseeds.cpp src/boosting.* src/rrset.* 
	${CXX} ${CPPFLAGS} src/main_moreseeds.cpp src/boosting.cpp src/getRSS.cpp \
		src/dsfmt/dSFMT.c -o bin/moreseeds

clean:
	rm -f obj/*
	rm -f bin/*

# For prrboost: LBONLY and LBTEST conflict with each other. Don't use them at the same time.
