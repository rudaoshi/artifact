#ifndef BACKPROP_NEURON_LAYER_H
#define BACKPROP_NEURON_LAYER_H

#include <smart_ptr>
using namespace std;

#include "neuron_layer.h"
#include <deeplearning/core/optimization/objective.h>
#include <deeplearning/core/machine/machine.h>

namespace deeplearning
{

	class bp_layer : public machine<VectorType, VectorType>,
									 public parameterirzed<VectorType>,
									 public objective<MatrixType, MatrixType>
	{
	protected:

		int input_dim;
		int output_dim;

		MatrixType W;
		VectorType b;

		MatrixType diff_W;
		VectorType diff_b;


		MatrixType input;
		MatrixType activation;
		MatrixType output;

		bool record_output;
		bool record_activation;

	public:


	  int get_input_dim();
		int get_output_dim();




		const MatrixType & get_W();
		const MatrixType & get_diff_W();

		const MatrixType & get_b();
		const MatrixType & get_diff_b();

		const MatrixType & get_activation();
		const MatrixType & get_output();


	public:
		bp_layer(int input_dim, int output_dim, bool record_output = true, bool record_activation = false);

		virtual void compute_gradient(const MatrixType & delta);

		virtual MatrixType compute_delta() = 0;

		virtual void backprop_delta(MatrixType & delta) = 0;
		

		virtual NumericType cost(const MatrixType & x) ;
		virtual MatrixType gradient(const MatrixType & x) ;



	};
}

#endif
