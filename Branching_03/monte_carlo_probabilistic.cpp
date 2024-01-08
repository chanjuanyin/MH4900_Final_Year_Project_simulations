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

double simulate_recursion(double t, double x_0, double a, double v) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(1.);
    std::uniform_real_distribution<> uniform_distribution(-1., 1.);
    std::uniform_real_distribution<> uniform_distribution_2(0., 1.);
    double tau_Nt = 0.;
    double X_tau_Nt = x_0;
    double output = 1.;
    double tau_diff = exponential_distribution(generator);
    while (tau_Nt + tau_diff < t) {
        X_tau_Nt += tau_diff * uniform_distribution(generator);
        tau_Nt += tau_diff;
        output *= tau_diff;

        // // ********** Not probabilistic **********
        // double V_1 = simulate_recursion(t - tau_Nt, X_tau_Nt, a, v); // Recursion branching happens here
        // double V_2 = simulate_recursion(t - tau_Nt, X_tau_Nt, a, v); // Recursion branching happens here
        // double V_3 = simulate_recursion(t - tau_Nt, X_tau_Nt, a, v); // Recursion branching happens here
        // output *= 3./2. * V_1 + 2. * V_2 * V_3;
        // output /= a;
        // // ********** Not probabilistic **********

        // ********** Probabilistic **********
        if (uniform_distribution_2(generator) <= (3. / 7.)) {
            double V_1 = simulate_recursion(t - tau_Nt, X_tau_Nt, a, v); // Recursion branching happens here
            output *= V_1;
        }
        else {
            double V_2 = simulate_recursion(t - tau_Nt, X_tau_Nt, a, v); // Recursion branching happens here
            double V_3 = simulate_recursion(t - tau_Nt, X_tau_Nt, a, v); // Recursion branching happens here
            output *= V_2 * V_3;
        }
        output *= 7. / (2. * a);
        // ********** Probabilistic **********

        tau_diff = exponential_distribution(generator);
    }
    output *= 2 * ( (1 + v) / ( pow((X_tau_Nt + t - tau_Nt), 2) - 4 ) + (1 - v) / ( pow((X_tau_Nt - t + tau_Nt), 2) - 4 ) );
    return output * exp(t);
}

void simulate_helper(xt::xarray<double> &arr, int start, int end, double t, double x_0, double a, double v, std::mutex& mtx) {
    int numSims = end - start;
    xt::xarray<double> local_arr = xt::zeros<double>({numSims});
    for (int i=0; i<numSims; i++) {
        local_arr[i] = simulate_recursion(t, x_0, a, v);
    }
    // Your critical section logic here
    {
        // Lock the mutex to protect the critical section
        std::lock_guard<std::mutex> lock(mtx);
        // Access and modify shared data
        xt::view(arr, xt::range(start, end)) = local_arr;
    }
}

double simulate(double t, double x_0, double a, double v, int total_sims, int numThreads) {
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
        threads.emplace_back(simulate_helper, std::ref(arr), start, end, t, x_0, a, v, std::ref(mtx));
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
        double estimated_value = simulate(1., 2., 1., sqrt(2), 1000000, 8);
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
    string directoryPath = "../Branching_03/output";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        std::__fs::filesystem::create_directory(directoryPath);
    }
    // find_optimal_numThreads();
    for (int i = 4; i <= 10; i++) {
        double x = static_cast<double>(i);
        int num_estimations = (i-3)*10+1;
        xt::xarray<double> arr = xt::zeros<double>({1, num_estimations});
        for (int j = 0; j < num_estimations; j++) {
            double t = static_cast<double>(j);
            t /= 10.;
            auto start_time = std::chrono::high_resolution_clock::now();
            double estimated_value = simulate(t, x, 1., sqrt(2), 100000, 6); // Optimal is 5 threads
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            cout << "x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
                << elapsed_time.count() << " seconds."<< endl;
            xt::view(arr, 0, j) = estimated_value;
            string file_name = "../Branching_03/output/x_equals_";
            file_name += to_string(i);
            // file_name += "_monte_carlo.csv"; // if not probabilistic
            file_name += "_monte_carlo_probabilistic.csv"; // if probabilistic
            std::ofstream out_file(file_name);
            xt::dump_csv(out_file, arr);
            cout << "x = "<< i << ", t = " << t << ", successfully exported to csv!" << endl;
        }
    }
    return 0;
}