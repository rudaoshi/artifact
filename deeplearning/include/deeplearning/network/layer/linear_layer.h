#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "backpropagation_layer.h"

namespace deeplearning
{
	class linear_layer: public backpropagation_layer
	{
	public:

		linear_layer(int input_dim, int output_dim);

		virtual MatrixType predict(const MatrixType & input);

		virtual VectorType predict(const VectorType & input);


        virtual MatrixType compute_delta(const MatrixType & input, const MatrixType & output);

        virtual MatrixType backprop_delta(const MatrixType & delta, const MatrixType & input, const MatrixType & output);



	};
}

#endif
