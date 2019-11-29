#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <typeinfo>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "cusparse_cholesky_solver.h"

// type alias
using Scalar = double;
using SparseMatrixCSC = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::ColMajor>;
using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::RowMajor>;
using Triplet = Eigen::Triplet<Scalar>;
using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using VectorI = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using Ordering = Eigen::AMDOrdering<SparseMatrixCSR::StorageIndex>;
using PermutationMatrix = Ordering::PermutationType;

static bool readMatrix(const std::string& filename, SparseMatrixCSC& A)
{
	std::ifstream ifs(filename);
	if (ifs.fail())
		return false;

	std::string line;
	while (std::getline(ifs, line))
	{
		if (line[0] != '%')
			break;
	}

	int n, m, nnz;
	std::sscanf(line.c_str(), "%d %d %d", &n, &m, &nnz);

	std::vector<Triplet> triplets;
	triplets.reserve(nnz);

	int nnzU = 0, nnzL = 0;
	while (std::getline(ifs, line))
	{
		int row, col;
		double value;
		std::sscanf(line.c_str(), "%d %d %lg", &row, &col, &value);

		triplets.push_back(Triplet(row - 1, col - 1, static_cast<Scalar>(value)));

		if (row < col)
			nnzU++;

		if (row > col)
			nnzL++;
	}

	if (triplets.size() != static_cast<size_t>(nnz))
		return false;

	A.resize(n, m);
	A.setFromTriplets(std::begin(triplets), std::end(triplets));

	// make A self-adjoint

	if (nnzU > 0 && nnzL == 0)
		A = SparseMatrixCSC(A.selfadjointView<Eigen::Upper>());

	if (nnzL > 0 && nnzU == 0)
		A = SparseMatrixCSC(A.selfadjointView<Eigen::Lower>());

	return true;
}

double residualInfinityNorm(const SparseMatrixCSC& A, const VectorR& b, const VectorR& x)
{
	const auto r = b - A * x;

	const double Ainf = VectorR::Map(A.valuePtr(), A.nonZeros()).lpNorm<Eigen::Infinity>();
	const double rinf = r.lpNorm<Eigen::Infinity>();
	const double xinf = x.lpNorm<Eigen::Infinity>();
	const double eps = 1e-15;

	return rinf / (Ainf * xinf + eps);
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: sample_large_cholesky nd6k.mtx [do-ordering]" << std::endl;
		return 0;
	}

	const bool doOrdering = argc > 2 ? (std::stoi(argv[2]) != 0) : true;

	////////////////////////////////////////////////////////////////////////////////////////////////
	// create system
	////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Reading Matrix ... ";

	SparseMatrixCSC Acsc;
	if (!readMatrix(argv[1], Acsc))
	{
		std::cerr << "Read failed." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::cout << "Done." << std::endl << std::endl;

	if (Acsc.rows() != Acsc.cols())
	{
		std::cerr << "Matrix must be square." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const int n = static_cast<int>(Acsc.rows());
	const int m = static_cast<int>(Acsc.cols());
	const int nnz = static_cast<int>(Acsc.nonZeros());

	std::cout << "=== Matrix : " << std::endl;
	std::cout << "Size       : " << n << " x " << m << std::endl;
	std::cout << "Non-zeros  : " << nnz << std::endl;
	std::cout << "Value Type : " << typeid(Scalar).name() << std::endl << std::endl;

	// generate x
	const VectorR x = VectorR::Random(n);

	// generate b
	const VectorR b = Acsc * x;

	////////////////////////////////////////////////////////////////////////////////////////////////
	// estimate x with Eigen
	////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Solving Ax = b with CPU ... ";

	Eigen::SimplicialLDLT<SparseMatrixCSC> cholesky;
	cholesky.analyzePattern(Acsc);

	const auto t0 = std::chrono::steady_clock::now();

	cholesky.factorize(Acsc);

	if (cholesky.info() != Eigen::Success)
	{
		std::cerr << "Factorize failed." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	VectorR xhatCPU = cholesky.solve(b);

	const auto t1 = std::chrono::steady_clock::now();
	std::cout << "Done." << std::endl << std::endl;

	////////////////////////////////////////////////////////////////////////////////////////////////
	// estimate x with cuSparse
	////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Solving Ax = b with GPU ... ";

	SparseMatrixCSR Acsr = Acsc; // solver supports CSR format
	auto solver = CuSparseCholeskySolver<Scalar>::create(n);

	if (doOrdering)
	{
		// compute permutation
		PermutationMatrix P;
		Ordering ordering;
		ordering(Acsr.selfadjointView<Eigen::Upper>(), P);

		// set permutation to solver
		solver->setPermutaion(n, P.indices().data());
	}

	solver->analyze(nnz, Acsr.outerIndexPtr(), Acsr.innerIndexPtr());

	VectorR xhatGPU(n);

	const auto t2 = std::chrono::steady_clock::now();

	solver->factorize(Acsr.valuePtr());

	if (solver->info() != CuSparseCholeskySolver<Scalar>::SUCCESS)
	{
		std::cerr << "Factorize failed." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	solver->solve(b.data(), xhatGPU.data());

	const auto t3 = std::chrono::steady_clock::now();
	std::cout << "Done." << std::endl << std::endl;

	////////////////////////////////////////////////////////////////////////////////////////////////
	// put summary
	////////////////////////////////////////////////////////////////////////////////////////////////
	const auto duration01 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	const auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

	std::cout << "=== Computation time : " << std::endl;
	std::printf("CPU : %7.3f [sec]\n", 1e-6 * duration01);
	std::printf("GPU : %7.3f [sec]\n", 1e-6 * duration23);
	std::cout << std::endl;

	std::cout << "=== Residual |b - A*x|/(|A|*|x|) : " << std::endl;
	std::printf("CPU : %e\n", residualInfinityNorm(Acsc, b, xhatCPU));
	std::printf("GPU : %e\n", residualInfinityNorm(Acsc, b, xhatGPU));

	return 0;
}
