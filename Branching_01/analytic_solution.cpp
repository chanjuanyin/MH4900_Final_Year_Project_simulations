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

xt::xarray<double> fix_t_varies_x(double t, double a, double v, double lower_bound, double upper_bound, double delta_x) {
    xt::xarray<double> arr = xt::arange(lower_bound, upper_bound + delta_x, delta_x);
    double double_1 = v * t;
    xt::xarray<double> factor_1 = {double_1};
    xt::xarray<double> arr2 = 6. / ( (arr + factor_1) * (arr + factor_1) );
    return arr2;
}

xt::xarray<double> fix_x_varies_t(double x, double a, double v, double lower_bound, double upper_bound, double delta_t_) {
    xt::xarray<double> arr = xt::arange(lower_bound, upper_bound + delta_t_, delta_t_);
    xt::xarray<double> arr2 = 6. / ( (x + v * arr) * (x + v * arr) );
    return arr2;
} 

int main()
{
    string directoryPath = "../Branching_01/output";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        std::__fs::filesystem::create_directory(directoryPath);
    }
    for (int i=0; i<=10; i++) {
        double t = static_cast<double> (i);
        xt::xarray<double> arr;
        if (i==0) {
            arr = fix_t_varies_x(t, 0.5, sqrt(2), 1., 10., 0.01);
            arr = arr.reshape({1, static_cast<int> ((10 - 1)/0.01) + 1});
        }
        else {
            arr = fix_t_varies_x(t, 0.5, sqrt(2), 0., 10., 0.01);
            arr = arr.reshape({1, static_cast<int> ((10 - 0)/0.01) + 1});
        }
        string file_name = "../Branching_01/output/t_equals_";
        file_name += to_string(i);
        file_name += ".csv";
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "t = "<< i << ", successfully exported to csv!" << endl;
    }
    for (int i=0; i<=10; i++) {
        double x = static_cast<double> (i);
        xt::xarray<double> arr;
        if (i==0) {
            arr = fix_x_varies_t(x, 0.3, sqrt(2), 1., 10., 0.01);
            arr = arr.reshape({1, static_cast<int> ((10 - 1)/0.01) + 1});
        }
        else {
            arr = fix_x_varies_t(x, 0.3, sqrt(2), 0., 10., 0.01);
            arr = arr.reshape({1, static_cast<int> ((10 - 0)/0.01) + 1});
        }
        string file_name = "../Branching_01/output/x_equals_";
        file_name += to_string(i);
        file_name += ".csv";
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "x = "<< i << ", successfully exported to csv!" << endl;
    }
    return 0;
}