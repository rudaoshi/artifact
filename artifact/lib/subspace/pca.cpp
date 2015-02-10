#include <liblearning/subspace/pca.h>
#include <cmath>
#include <algorithm>
using namespace subspace;

#include <Eigen/Dense>


pca::pca(const MatrixType & train_)
{
	

	MatrixType centered_train = train_;

	mean = centered_train.colwise().sum();

	centered_train.colwise() -= mean;

	int d = centered_train.rows();
	int n = centered_train.cols();

	centered_train /= std::sqrt(NumericType(n));


	if (train_.rows() <= train_.cols())
		pca_std(centered_train);
	else
		pca_trans(centered_train);



}


void pca::pca_auto(const MatrixType & centered_train)
{
	int d = centered_train.rows();
	int n = centered_train.cols();

	if (d <= n)
	{
		pca_std(centered_train);
	}
	else
	{
		pca_trans(centered_train);
	}
}


void pca::pca_std(const MatrixType & centered_train_)
{
	MatrixType C = centered_train_ * centered_train_.transpose();
	C = 0.5*(C+C.transpose());

	Eigen::SelfAdjointEigenSolver<MatrixType> eigensolver(C);
    eigenval = eigensolver.eigenvalues();
    P = eigensolver.eigenvectors();

}

void pca::pca_trans(const MatrixType & centered_train_)
{
	MatrixType Ct = centered_train_.transpose() * centered_train_;
	Ct = 0.5*(Ct+Ct.transpose());

	Eigen::SelfAdjointEigenSolver<MatrixType> eigensolver(Ct);
    eigenval = eigensolver.eigenvalues();
    P = centered_train_*eigensolver.eigenvectors();

	VectorType eigv_norm = P.colwise().norm();

	for (int i = 0;i<P.cols();i ++)
	{
		P.col(i).array() /= std::max(std::numeric_limits<NumericType>::epsilon(),eigv_norm(i));
	}


}

MatrixType pca::apply(const MatrixType & test, int k)
{
	MatrixType centered_test = test;
	centered_test.colwise() -= mean;

	return P.block(0,P.cols()-k,P.rows(),k).transpose()*test;

}
