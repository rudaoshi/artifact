#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "bp_layer.h"

namespace deeplearning
{
	class linear_layer: public mlp_layer
	{
	public:

		linear_layer（ int input_dim, int output_dim, bool record_output_ = true, bool record_activation_ = false);

		virtual MatrixType output(const MatrixType & input);

		virtual MatrixType delta(const MatrixType & output, const MatrixType & error_diff);

		virtual void delta_backprop(MatrixType & delta, const MatrixType & input);

	};
}

#endif
