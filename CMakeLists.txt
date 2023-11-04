cmake_minimum_required(VERSION 3.18..3.20)

project(mymodule)

# find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
# find_package(pybind11 REQUIRED CONFIG)
# find_package(xtensor-python REQUIRED)
find_package(xtensor REQUIRED)

add_executable(sample_1 sample_1.cpp)
add_executable(telegraph_equation_analytic_solution telegraph_equation/analytic_solution.cpp)
add_executable(telegraph_equation_monte_carlo_simulation telegraph_equation/monte_carlo_simulation.cpp)

target_link_libraries(sample_1 PUBLIC xtensor)
target_link_libraries(telegraph_equation_analytic_solution PUBLIC xtensor)
target_link_libraries(telegraph_equation_monte_carlo_simulation PUBLIC xtensor)
