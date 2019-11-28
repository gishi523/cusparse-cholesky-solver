#include <iostream>
#include <fstream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cusparse_cholesky_solver.h>

// type alias
using Scalar = double;
using SparseMatrixCSC = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::ColMajor>;
using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::RowMajor>;
using Triplet = Eigen::Triplet<Scalar>;
using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

std::vector<Triplet> getnerateTriplets(int n, Scalar diagVal = 4, Scalar nonDiagVal = -1)
{
	std::vector<Triplet> triplets;
	for (int i = 0; i < n; i++)
	{
		triplets.push_back(Triplet(i, i, diagVal));

		if (i > 0)
			triplets.push_back(Triplet(i, i - 1, nonDiagVal));

		if (i < n - 1)
			triplets.push_back(Triplet(i, i + 1, nonDiagVal));
	}
	return triplets;
}

int main()
{
	constexpr int n = 64;
	const auto triplets = getnerateTriplets(n, 4, -1);
	const int nnz = static_cast<int>(triplets.size());

	// generate symmetric sparse matrix
	SparseMatrixCSC Acsc(n, n);
	Acsc.setFromTriplets(std::begin(triplets), std::end(triplets));

	// generate x
	const VectorR x = VectorR::Random(n);

	// generate b
	const VectorR b = Acsc * x;

	// estimate x with Eigen
	Eigen::SimplicialLDLT<SparseMatrixCSC> cholesky;
	cholesky.analyzePattern(Acsc);
	cholesky.factorize(Acsc);
	VectorR xhatCPU = cholesky.solve(b);

	// estimate x with cuSparse
	SparseMatrixCSR Acsr = Acsc; // solver supports CSR format
	auto solver = CuSparseCholeskySolver<Scalar>::create(n);
	solver->analyze(nnz, Acsr.outerIndexPtr(), Acsr.innerIndexPtr());

	VectorR xhatGPU(n);
	solver->factorize(Acsr.valuePtr());
	solver->solve(b.data(), xhatGPU.data());

	// put errros
	std::cout << "error CPU: " << (x - xhatCPU).squaredNorm() << std::endl;
	std::cout << "error GPU: " << (x - xhatGPU).squaredNorm() << std::endl;

	return 0;
}
