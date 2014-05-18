#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "bp_layer.h"

namespace deeplearning
{
	class linear_layer: public bp_layer
	{
	public:

		linear_layer（ int input_dim, int output_dim, bool record_output_ = true, bool record_activation_ = false);

		virtual MatrixType predict(const MatrixType & input);

		virtual VectorType predict(const VectorType & input);

		virtual void compute_gradient(const MatrixType & delta);

		virtual MatrixType compute_delta() = 0;

		virtual void backprop_delta(MatrixType & delta) = 0;

	};
}

#endif
