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

void simulate_helper(xt::xarray<double> &arr_t_tau_Nt, xt::xarray<double> &arr_X_tau_Nt, xt::xarray<double> &arr_product, 
        int start, int end, double x_0, double t, double a, double v, std::mutex& mtx) {
    int numSims = end - start;
    xt::xarray<double> local_arr_1 = xt::zeros<double>({numSims});
    xt::xarray<double> local_arr_2 = xt::zeros<double>({numSims});
    xt::xarray<double> local_arr_3 = xt::zeros<double>({numSims});
    int count = 0;
    double tau_Nt = 0.;
    double X_tau_Nt = x_0;
    double product_1 = 1.;
    while (count<numSims) {
        xt::xarray<double> random_exponential_array = xt::random::exponential({numSims}, 1.); // rate one poisson process (with every jump interval (\tau_i - \tau_{i-1}) modelled by exponential distribution with \lambda = 1 )
        xt::xarray<double> random_uniform_array = xt::random::rand({numSims}, -1., 1.); // uniformly drawn from -1 to 1
        for (int i=0; i<numSims; i++) {
            if (tau_Nt + random_exponential_array[i] < t) {
                X_tau_Nt += random_exponential_array[i] * random_uniform_array[i];
                product_1 *= (a * a) * random_exponential_array[i];
                tau_Nt += random_exponential_array[i];
            }
            else {
                local_arr_1[count] = t - tau_Nt;
                local_arr_2[count] = X_tau_Nt;
                local_arr_3[count] = product_1;
                tau_Nt = 0.;
                X_tau_Nt = x_0;
                product_1 = 1.;
                count++;
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
        xt::view(arr_t_tau_Nt, xt::range(start, end)) = local_arr_1;
        xt::view(arr_X_tau_Nt, xt::range(start, end)) = local_arr_2;
        xt::view(arr_product, xt::range(start, end)) = local_arr_3;
    }
}

double simulate(double x_0, double t, double a, double v, int total_sims, int numThreads) {
    xt::xarray<double> arr_t_tau_Nt = xt::zeros<double>({total_sims});
    xt::xarray<double> arr_X_tau_Nt = xt::zeros<double>({total_sims});
    xt::xarray<double> arr_product = xt::zeros<double>({total_sims});
    xt::xarray<double> arr_w = xt::zeros<double>({total_sims});
    xt::xarray<double> arr_outcome = xt::zeros<double>({total_sims});
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
        threads.emplace_back(simulate_helper, std::ref(arr_t_tau_Nt), std::ref(arr_X_tau_Nt), std::ref(arr_product), 
            start, end, x_0, t, a, v, std::ref(mtx));
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    double double_1 = sqrt(a*a + v*v);
    xt::xarray<double> factor_1 = {double_1};
    double double_2 = sqrt(a*a + v*v) * a / v;
    xt::xarray<double> factor_2 = {double_2};

    arr_w = factor_1 * ( xt::exp((arr_X_tau_Nt + arr_t_tau_Nt) * v) + xt::exp((-arr_X_tau_Nt - arr_t_tau_Nt) * v)
        + xt::exp((arr_X_tau_Nt - arr_t_tau_Nt) * v) + xt::exp((-arr_X_tau_Nt + arr_t_tau_Nt) * v) ) + 
        factor_2 * ( xt::exp((arr_X_tau_Nt + arr_t_tau_Nt) * v) - xt::exp((arr_X_tau_Nt - arr_t_tau_Nt) * v)
        - xt::exp((-arr_X_tau_Nt - arr_t_tau_Nt) * v) + xt::exp((-arr_X_tau_Nt + arr_t_tau_Nt) * v) );

    // arr_w = factor_1 * ( xt::exp((arr_X_tau_Nt + arr_t_tau_Nt) * v) + xt::exp((-arr_X_tau_Nt - arr_t_tau_Nt) * v) );
    
    arr_outcome = arr_w * arr_product;
    double avg = xt::average(arr_outcome)[0];
    return avg * exp(t);
}

void find_optimal_numThreads() // Optimal should be 6 threads
{
    for (int numThreads = 1; numThreads <= 20; numThreads++) {
        // Record the start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run the Monte Carlo simulation
        double estimated_value = simulate(1, 1, 0.3, 0.3, 1000000, 8);
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
    // find_optimal_numThreads();
    xt::xarray<double> arr = xt::zeros<double>({11, 11});
    for (int i = -5; i <= 5; i++) {
        for (int j = 0; j <= 10; j++) {
            double x = static_cast<double>(i);
            double t = static_cast<double>(j);
            auto start_time = std::chrono::high_resolution_clock::now();
            double estimated_value = simulate(x, t, 0.3, 0.3, 1000000, 6); // Optimal is 6 threads
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            cout << "x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
                << elapsed_time.count() << " seconds."<< endl;
            xt::view(arr, i+5, j) = estimated_value;
        }
    }
    string directoryPath = "../Dalang/output";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        std::__fs::filesystem::create_directory(directoryPath);
    }
    string file_name = "../Dalang/output/estimated_values.csv";
    std::ofstream out_file(file_name);
    xt::dump_csv(out_file, arr);
    cout << "Successfully exported to csv!" << endl;
    return 0;
}