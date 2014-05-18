#include <liblearning/kernel/linear_kernel.h>

using namespace kernelmethod;

linear_kernel::linear_kernel(void)
{
}


linear_kernel::~linear_kernel(void)
{
}


NumericType linear_kernel::eval(const VectorType & x, const VectorType & y) const
{
	return x.dot(y);
}

MatrixType linear_kernel::eval(const MatrixType & X, const MatrixType & Y) const
{
	return X.transpose()*Y;
}
