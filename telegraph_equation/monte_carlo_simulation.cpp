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

void simulate_helper(xt::xarray<double> &arr, int start, int end, double x_0, double t, double a, double v, std::mutex& mtx) {
    int numSims = end - start;
    xt::xarray<double> local_arr = xt::zeros<double>({numSims});
    int count = 0;
    double integral_sum = 0.;
    double absolute_sum = 0.;
    bool sgn = true;
    while (count<numSims) {
        xt::xarray<double> random_array = xt::random::exponential({numSims}, a); // poisson process with every jump interval (\tau_i - \tau_{i-1}) modelled by exponential distribution with \lambda = a
        for (int i=0; i<numSims; i++) {
            absolute_sum += random_array[i];
            if (sgn) {
                integral_sum += random_array[i];
                sgn = false;
            }
            else {
                integral_sum -= random_array[i];
                sgn = true;
            }
            if (absolute_sum >= t) {
                if (sgn) {
                    integral_sum += (absolute_sum - t);
                }
                else {
                    integral_sum -= (absolute_sum - t);
                }
                local_arr[count++] = integral_sum;
                absolute_sum = 0.;
                integral_sum = 0.;
                sgn = true;
                if (count==numSims) {
                    break;
                }
            }
        }
    }
    // Your critical section logic here
    {
        // Lock the mutex to protect the critical section
        std::lock_guard<std::mutex> lock(mtx);
        // Access and modify shared data
        xt::view(arr, xt::range(start, end)) = local_arr;
    }
}

double simulate(double x_0, double t, double a, double v, int total_sims, int numThreads) {
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
        threads.emplace_back(simulate_helper, std::ref(arr), start, end, x_0, t, a, v, std::ref(mtx));
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    double double_1 = sqrt(a*a + v*v);
    xt::xarray<double> factor_1 = {double_1};
    arr = factor_1 * ( xt::exp(-(x_0 + v * arr)) + xt::exp(x_0 + v * arr) + xt::exp(-(x_0 - v * arr)) + xt::exp(x_0 - v * arr));
    double avg = xt::average(arr)[0];
    return avg;
}

void find_optimal_numThreads() // Optimal should be 8 threads
{
    for (int numThreads = 1; numThreads <= 20; numThreads++) {
        // Record the start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run the Monte Carlo simulation
        double estimated_value = simulate(1, 1, 0.5, 0.5, 1000000, 8);
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
    xt::xarray<double> arr = xt::zeros<double>({11, 11});
    for (int i = -5; i <= 5; i++) {
        for (int j = 0; j <= 10; j++) {
            double x = static_cast<double>(i);
            double t = static_cast<double>(j);
            auto start_time = std::chrono::high_resolution_clock::now();
            double estimated_value = simulate(x, t, 0.5, 0.5, 1000000, 8); // Optimal is 8 threads
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            cout << "x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
                << elapsed_time.count() << " seconds."<< endl;
            xt::view(arr, i+5, j) = estimated_value;
        }
    }
    string directoryPath = "../telegraph_equation/output";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        std::__fs::filesystem::create_directory(directoryPath);
    }
    string file_name = "../telegraph_equation/output/estimated_values.csv";
    std::ofstream out_file(file_name);
    xt::dump_csv(out_file, arr);
    cout << "Successfully exported to csv!" << endl;
    return 0;
}