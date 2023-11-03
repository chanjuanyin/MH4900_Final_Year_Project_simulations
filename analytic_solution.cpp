#include <bits/stdc++.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <istream>
#include <fstream>
#include <xtensor/xarray.hpp>
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
    double double_1 = a + sqrt(a*a + v*v);
    xt::xarray<double> factor_1 = {double_1};
    double double_2 = -a + sqrt(a*a + v*v);
    xt::xarray<double> factor_2 = {double_2};
    xt::xarray<double> arr2 = (xt::exp(arr) + xt::exp(-arr)) * (factor_1 * xt::exp(factor_2 * t) + factor_2 * xt::exp(-factor_1 * t));
    return arr2;
}

xt::xarray<double> fix_x_varies_t(double x, double a, double v, double lower_bound, double upper_bound, double delta_t_) {
    xt::xarray<double> arr = xt::arange(lower_bound, upper_bound + delta_t_, delta_t_);
    double double_1 = a + sqrt(a*a + v*v);
    xt::xarray<double> factor_1 = {double_1};
    double double_2 = -a + sqrt(a*a + v*v);
    xt::xarray<double> factor_2 = {double_2};
    double double_3 = exp(x) + exp(-x);
    xt::xarray<double> factor_3 = {double_3};
    xt::xarray<double> arr2 = factor_3 * (factor_1 * xt::exp(factor_2 * arr) + factor_2 * xt::exp(-factor_1 * arr));
    return arr2;
} 

int main()
{
    for (int i=0; i<=10; i++) {
        double t = static_cast<double> (i);
        xt::xarray<double> arr = fix_t_varies_x(t, 0.5, 0.5, -5, 5, 0.01);
        arr = arr.reshape({1, static_cast<int> ((5 + 5)/0.01) + 1});
        // cout << "arr = { ";
        // for (double d : arr) {
        //   cout << d << ", ";
        // }
        // cout << "}";
        // cout << arr << endl;
        string file_name = "../output/t_equals_";
        file_name += to_string(i);
        file_name += ".csv";
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "t = "<< i << ", successfully exported to csv!" << endl;
    }
    for (int i=-5; i<=5; i++) {
        double x = static_cast<double> (i);
        xt::xarray<double> arr = fix_x_varies_t(x, 0.5, 0.5, 0, 10, 0.01);
        arr = arr.reshape({1, static_cast<int> ((5 + 5)/0.01) + 1});
        // cout << "arr = { ";
        // for (double d : arr) {
        //   cout << d << ", ";
        // }
        // cout << "}";
        // cout << arr << endl;
        string file_name = "../output/x_equals_";
        if (i<0) {
            file_name += "neg_";
            file_name += to_string(abs(i));
        }
        else if (i==0) {
            file_name += "0";
        }
        else {
            file_name += "pos_";
            file_name += to_string(i);
        }
        file_name += ".csv";
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "x = "<< i << ", successfully exported to csv!" << endl;
    }
    return 0;
}