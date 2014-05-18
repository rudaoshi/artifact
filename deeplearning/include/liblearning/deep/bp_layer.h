#ifndef BACKPROP_NEURON_LAYER_H
#define BACKPROP_NEURON_LAYER_H

#include "neuron_layer.h"


namespace deeplearning
{
	class bp_layer : public neuron_layer
	{
	protected:

		int input_dim;
		int output_dim;

		MatrixType W;
		VectorType b;

		MatrixType diff_W;
		VectorType diff_b;

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

		virtual void backprop_diff(const MatrixType & input, const MatrixType & delta);

		virtual MatrixType delta(const MatrixType & output, const MatrixType & error_diff) = 0;

		virtual void delta_backprop(MatrixType & delta, const MatrixType & input) = 0;


	};
}

#endif
