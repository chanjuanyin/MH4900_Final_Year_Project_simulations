#include <bits/stdc++.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <istream>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>
// #include <xtensor/xjson.hpp>

using namespace std;
typedef long long ll;
typedef pair<int,int> PP;
typedef double ld;
const double eps=1e-6;

double simulate_recursion(double t, double x_0, double y_0, double a, double v) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(1.);
    std::uniform_real_distribution<> radiusDistribution(0., 1.);
    std::uniform_real_distribution<> angleDistribution(0., 360.);
    double tau_Nt = 0.;
    double X_tau_Nt = x_0;
    double Y_tau_Nt = y_0;
    double output = 1.;
    double tau_diff = exponential_distribution(generator);
    while (tau_Nt + tau_diff < t) {
        double angle  = angleDistribution(generator);
        double radius = radiusDistribution(generator);
        radius = (pow(1 - pow(1 - radius, 2.0), 0.5)) * tau_diff;
        double x = radius * sin(angle);
        double y = radius * cos(angle);
        X_tau_Nt += x;
        Y_tau_Nt += y;
        tau_Nt += tau_diff;
        output *= tau_diff;

        // ********** Independently generating branches then multiply to the output **********
        double V_1 = simulate_recursion(t - tau_Nt, X_tau_Nt, Y_tau_Nt, a, v); // Recursion branching happens here
        output *= V_1;
        // ********** Independently generating branches then multiply to the output **********

        output /= a;
        tau_diff = exponential_distribution(generator);
    }
    double t_minus_tau_Nt = t - tau_Nt;
    output *= 6 * a * (2 * pow(t_minus_tau_Nt, 2) + pow(X_tau_Nt + Y_tau_Nt, 2)) / pow(2 * pow(t_minus_tau_Nt, 2) - pow(X_tau_Nt + Y_tau_Nt, 2), 2) 
              - 12 * sqrt(3) * a * t_minus_tau_Nt * (X_tau_Nt + Y_tau_Nt) / pow(2 * pow(t_minus_tau_Nt, 2) - pow(X_tau_Nt + Y_tau_Nt, 2), 2);
    return output * exp(t);
}

void simulate_helper(xt::xarray<double> &arr, int start, int end, double t, double x_0, double y_0, double a, double v, std::mutex& mtx) {
    int numSims = end - start;
    xt::xarray<double> local_arr = xt::zeros<double>({numSims});
    for (int i=0; i<numSims; i++) {
        local_arr[i] = simulate_recursion(t, x_0, y_0, a, v);
    }
    // Your critical section logic here
    {
        // Lock the mutex to protect the critical section
        std::lock_guard<std::mutex> lock(mtx);
        // Access and modify shared data
        xt::view(arr, xt::range(start, end)) = local_arr;
    }
}

double simulate(double t, double x_0, double y_0, double a, double v, int total_sims, int numThreads) {
    xt::xarray<double> arr = xt::zeros<double>({total_sims});
    std::vector<std::thread> threads;
    std::mutex mtx; // Create a mutex
    int chunk_size = total_sims / numThreads;
    
    for (int i=0; i<numThreads; i++) {
        int start = i * chunk_size;
        int end;
        if (i<(numThreads-1)) {
            end = (i+1) * chunk_size;
        }
        else {
            end = total_sims;
        }
        threads.emplace_back(simulate_helper, std::ref(arr), start, end, t, x_0, y_0, a, v, std::ref(mtx));
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    double avg = xt::average(arr)[0];
    return avg;
}

void find_optimal_numThreads() // Optimal should be 5 threads
{
    for (int numThreads = 1; numThreads <= 20; numThreads++) {
        // Record the start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run the Monte Carlo simulation
        double estimated_value = simulate(1.,4., 4., 1., sqrt(3), 1000000, 8);
        cout << estimated_value << endl;

        // Record the end time
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate the elapsed time
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "numThreads = " << numThreads << ", Execution time: " << elapsed_time.count() << " seconds" << endl;
    }
}

int main()
{
    string directoryPath = "../Branching_04_corrected/output";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        std::__fs::filesystem::create_directory(directoryPath);
    }
    // find_optimal_numThreads();
    for (int i = 4; i <= 6; i++) {
        for (int j = 4; j <= 6; j++) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(j);
            int num_estimations = 31;
            xt::xarray<double> arr = xt::zeros<double>({1, num_estimations});
            for (int k = 0; k < num_estimations; k++) {
                double t = static_cast<double>(k);
                t /= 10.;
                auto start_time = std::chrono::high_resolution_clock::now();
                double estimated_value = simulate(t, x, y, 1., sqrt(3), 100000, 6); // Optimal is 5 threads
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_time = end_time - start_time;
                cout << "x = " << x << ", y = " << y << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
                    << elapsed_time.count() << " seconds."<< endl;
                xt::view(arr, 0, k) = estimated_value;
                string file_name = "../Branching_04_corrected/output/x_equals_";
                file_name += to_string(i);
                file_name += "_y_equals_";
                file_name += to_string(j);
                file_name += "_monte_carlo.csv";
                std::ofstream out_file(file_name);
                xt::dump_csv(out_file, arr);
                cout << "x = "<< i << ", y = " << j << ", t = " << t << ", successfully exported to csv!" << endl;
            }
        }
    }
    return 0;
}