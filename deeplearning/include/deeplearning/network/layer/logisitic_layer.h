#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H


#include "bp_layer.h"

namespace deeplearning
{
	class logistic_layer: public bp_layer
	{
	public:

		logistic_layer(int input_dim, int output_dim, bool record_output = true, bool record_activation = false);

		virtual MatrixType predict(const MatrixType & input);

		virtual VectorType predict(const VectorType & input);

		virtual void compute_gradient(const MatrixType & delta);

		virtual MatrixType compute_delta() = 0;

		virtual void backprop_delta(MatrixType & delta) = 0;

	};
}

#endif
