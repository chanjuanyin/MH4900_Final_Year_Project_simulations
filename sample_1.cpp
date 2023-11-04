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
 
// const int LENGTH=105;
// double a[10];
// int dcmp(double x) {
//     if(fabs(x)<eps) return 0;
//     else return x<0? -1:1;
// }

int main()
{
    cout <<"Hello World" << endl;
    cout <<"Hello World" << endl;
    cout <<"Hello World" << endl;
    xt::xarray<int> arr
      {1, 2, 3, 4, 5, 6, 7, 8, 9};
    arr.reshape({3, 3});
    cout << arr << endl;
    xt::xarray<double> arr2 = xt::ones<double>({3,4}) * 0.5;
    cout << arr2 << endl;
    std::system("cd ..; mkdir haha");
    string directoryPath = "../haha2";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        // Directory does not exist, so create it
        if (std::__fs::filesystem::create_directory(directoryPath)) {
            std::cout << "Directory created successfully." << std::endl;
        } else {
            std::cerr << "Failed to create directory." << std::endl;
        }
    } else {
        std::cout << "Directory already exists." << std::endl;
    }
    return 0;
}