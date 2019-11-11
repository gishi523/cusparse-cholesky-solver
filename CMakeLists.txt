cmake_minimum_required(VERSION 3.1)

project(cuSparseCholeskySolver)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall")
endif()

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB srcs ./*.cpp ./*.h)
cuda_add_executable(cusparse_cholesky_solver ${srcs})
target_link_libraries(cusparse_cholesky_solver ${CUDA_cusparse_LIBRARY})
