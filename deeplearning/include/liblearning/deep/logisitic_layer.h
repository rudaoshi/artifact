#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H


#include "bp_layer.h"

namespace deeplearning
{
	class logistic_layer: public bp_layer
	{
	public:
		virtual MatrixType output(const MatrixType & input);

		virtual MatrixType delta(const MatrixType & output, const MatrixType & error_diff);

		virtual void delta_backprop(MatrixType & delta, const MatrixType & input);

	};
}

#endif
