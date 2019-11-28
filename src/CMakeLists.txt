cmake_minimum_required(VERSION 3.1)

project(cusparse_cholesky_solver)

include_directories(../include)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall")
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB srcs ./*.cpp ./*.h)
cuda_add_library(cusparse_cholesky_solver ${srcs})
target_link_libraries(cusparse_cholesky_solver ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})
