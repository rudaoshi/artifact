#ifndef KERNEL_H
#define KERNEL_H


#include <liblearning/core/config.h>

namespace kernelmethod
{
	class kernel 
	{

	public:
		virtual NumericType eval(const VectorType & x, const VectorType & y) const = 0 ;

		virtual MatrixType eval(const MatrixType & X, const MatrixType & Y) const = 0  ;

	};
}


#endif