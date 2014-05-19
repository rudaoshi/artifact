#ifndef BACKPROP_NEURON_LAYER_H
#define BACKPROP_NEURON_LAYER_H


#include <deeplearning/core/config.h>
#include <deeplearning/core/optimization/objective.h>
#include <deeplearning/core/machine/machine.h>

namespace deeplearning
{

	class backpropagation_layer : public machine<VectorType, VectorType>,
									 public parameterized<VectorType>,
									 public objective<MatrixType, MatrixType>
	{
	protected:

		int input_dim;
		int output_dim;

		MatrixType W;
		VectorType b;

	public:


	    int get_input_dim();
		int get_output_dim();


	public:
		backpropagation_layer(int input_dim, int output_dim);

		virtual VectorType compute_param_gradient(const MatrixType & delta, const MatrixType & input, const MatrixType & output);

		virtual MatrixType compute_delta(const MatrixType & input, const MatrixType & output) = 0;

		virtual MatrixType backprop_delta(const MatrixType & delta, const MatrixType & input, const MatrixType & output) = 0;

        virtual const VectorType  & get_parameter();
        virtual void set_parameter(const VectorType & parameter_);
		

		virtual NumericType cost(const MatrixType & x) ;
		virtual MatrixType cost_gradient(const MatrixType & x) ;



	};
}

#endif
