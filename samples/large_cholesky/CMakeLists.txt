cmake_minimum_required(VERSION 3.1)

project(sample_large_cholesky)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall")
endif()

include_directories(../../include)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB srcs ./*.cpp ./*.h)
cuda_add_executable(sample_large_cholesky ${srcs})
target_link_libraries(sample_large_cholesky cusparse_cholesky_solver)
