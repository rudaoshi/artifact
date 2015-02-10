/*
 * restricted_boltzmann_machine.h
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */

#ifndef RESTRICTED_BOLTZMANN_MACHINE_H_
#define RESTRICTED_BOLTZMANN_MACHINE_H_

#include <liblearning/core/dataset.h>

#include "neuron_type.h"



namespace deep
{
	using namespace core;
	class restricted_boltzmann_machine
	{
		int numvis;
		int numhid;

		MatrixType W;
		VectorType b;
		VectorType c;

		neuron_type type;

		std::vector<double> train_error;

		MatrixType W_inc;
		VectorType c_inc;
		VectorType b_inc;

		// set convergence values
		NumericType epsilonw; // learning rate for weights
		NumericType epsilonb; // learning rate for biases of visible units
		NumericType epsilonc; // learning rate for biases of hidden units
		NumericType weightcost;

		NumericType initialmomentum;
		NumericType finalmomentum;

		NumericType cur_momentum;

		int batch_size;

		int iter_per_batch;


	private:

		MatrixType output(const MatrixType & data);

		NumericType train_one_step(const MatrixType & X0);

	public:
		restricted_boltzmann_machine(int numvis_, int numhid_, neuron_type type_);

		virtual ~restricted_boltzmann_machine(void);

		void set_batch_setting(int batch_size, int iter_per_batch);

		NumericType train(const shared_ptr<dataset> & X, int num_iter);

		shared_ptr<dataset> output(const dataset & X);

//		NumericType train_batch(const shared_ptr<dataset> & X, int batch_size, int iter_per_batch, int num_iter);

		shared_ptr<dataset> output_batch(const dataset & X, int batch_size);

		MatrixType & get_W();
		VectorType & get_b();
		VectorType & get_c();

	};
}

#endif /* RESTRICTED_BOLTZMANN_MACHINE_H_ */
