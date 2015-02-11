#ifndef LAYERWISE_INITIALIZER_H
#define LAYERWISE_INITIALIZER_H

#include <liblearning/core/dataset.h>
#include "neuron_type.h"


namespace deep
{
	using namespace core;
	class layerwise_initializer
	{

	public:

		virtual void init(int input_dim, int output_dim, neuron_type type_) = 0;

		virtual NumericType train(const shared_ptr<dataset> & train_data) = 0;

		virtual shared_ptr<dataset> get_output() = 0;

		virtual MatrixType get_W1() = 0;
		virtual VectorType get_b1() = 0;

		virtual MatrixType get_W2() = 0;
		virtual VectorType get_b2() = 0;

		virtual layerwise_initializer* clone() = 0;


	};
}

#endif
