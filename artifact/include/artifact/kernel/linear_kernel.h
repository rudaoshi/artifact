#ifndef LINEAR_KERNEL_H_
#define LINEAR_KERNEL_H_

#include <liblearning/core/config.h>
#include "kernel.h"

namespace kernelmethod
{
	class linear_kernel: public kernel
	{
		public:
			linear_kernel(void);
			~linear_kernel(void);

			virtual NumericType eval(const VectorType & x, const VectorType & y) const;

			virtual MatrixType eval(const MatrixType & X, const MatrixType & Y) const ;

	};
	
}


#endif
