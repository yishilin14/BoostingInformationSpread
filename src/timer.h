#ifndef TIMER_H_
#define TIMER_H_

#include <stdio.h>

#include <chrono>

class Timer {
    std::chrono::time_point<std::chrono::system_clock> start_time;

   public:
    Timer() : start_time(std::chrono::system_clock::now()) {}

    void Reset() { start_time = std::chrono::system_clock::now(); }

    void PrintTimeElapsed() const {
        printf("Time elapsed: %.2lfs\n", TimeElapsed());
    }

    double TimeElapsed() const {
        std::chrono::duration<double> elapsed_seconds =
            std::chrono::system_clock::now() - start_time;
        return elapsed_seconds.count();
    }

    static void PrintCurrentTime(const char* const message = "") {
        std::time_t end_time = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        printf("%s: %s\n", message, std::ctime(&end_time));
    }
};

#endif
