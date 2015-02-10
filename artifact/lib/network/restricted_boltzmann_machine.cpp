/*
* restricted_boltzmann_machine.cpp
*
*  Created on: 2010-6-4
*      Author: sun
*/

#include <liblearning/deep/restricted_boltzmann_machine.h>
#include <liblearning/core/data_splitter.h>

#include <liblearning/util/matrix_util.h>
#include <liblearning/util/math_util.h>

#include <liblearning/deep/neuron_layer_operation.h>

#include <iostream>

using namespace deep;



restricted_boltzmann_machine::restricted_boltzmann_machine(int numvis_, int numhid_, neuron_type type_):	numvis(numvis_), numhid(numhid_), type(type_)
{

	if (type == linear)
	{
		epsilonw = .001;
		epsilonb = .001;
		epsilonc = .001;
		weightcost = .0002;

	}
	else if (type == logistic)
	{
		epsilonw = .1;
		epsilonb = .1;
		epsilonc = .1;
		weightcost = .0002;

	}

	initialmomentum = 0.5;
	finalmomentum = 0.9;

		// initialize weights and biases
	W = (0.1*randn<MatrixType>(numhid,numvis));

	b.setZero(numvis);
	c.setZero(numhid);
}

restricted_boltzmann_machine::~restricted_boltzmann_machine()
{

}

void restricted_boltzmann_machine::set_batch_setting(int batch_size_, int iter_per_batch_)
{
	batch_size = batch_size;
	iter_per_batch = iter_per_batch_;
}

NumericType restricted_boltzmann_machine::train_one_step(const MatrixType & X0)
{

	// constrastive divergence goes from X0 -> Y0 -> X1 -> Y1 to obtain
	// 1 step in a Markov chain.  From this chain, the updates are calculated
	// to minimize energy, according to the formula described in (Hinton
	// & Salakhutdinov, 2006).




	int N = X0.cols();

	// X0 -> Y0
	/*MatrixType Y0(numhid, N);

	if (type == linear)
	{
	//do linear level
	linear_transform(Y0.data(), numhid, numvis, N, 1.0, W.data(), 'N',
	X0.data(), 1, c.data());

	}
	else
	{
	// do logistic unit levels
	logistic_transform(Y0.data(), numhid, numvis, N, -1.0, W.data(), 'N',
	X0.data(), -1, c.data());
	}*/

	MatrixType Y0 = output(X0);

	// sample from forward mapped values (treated as probabilities)
	// sampling here prevents overtraining in this stage
	MatrixType Y0_bool(numhid, N);
	if (type == linear)
		Y0_bool = Y0 + randn<MatrixType>(numhid, N);
	else
		Y0_bool = Y0 > rand<MatrixType>(numhid, N);

	// Y0 -> X1
	MatrixType X1;

	logistic_layer_output(X1, W, 'T', Y0_bool, b);

	// X1 -> Y1
	/*MatrixType Y1(numhid, N);

	if (type == linear)
	{
	//do linear level
	linear_transform(Y1.data(), numhid, numvis, N, 1.0, W.data(), 'N',
	X1.data(), 1, c.data());

	}
	else
	{
	// do logistic unit levels
	logistic_transform(Y1.data(), numhid, numvis, N, -1.0, W.data(), 'N',
	X1.data(), -1, c.data());
	}
	*/
	MatrixType Y1 = output(X1);


	// compute reconstruction error
	NumericType err = (X0-X1).squaredNorm();



	// update weights and biases
	rbm_W_update(W_inc, X0,Y0,X1,Y1,W,cur_momentum,epsilonw, weightcost);
//	W_inc = cur_momentum*W_inc + 	epsilonw*( (Y0*X0.transpose()-Y1*X1.transpose())/N - weightcost*W);
	b_inc = cur_momentum * b_inc + (epsilonb / N) * (X0.rowwise().sum()- X1.rowwise().sum());
	c_inc = cur_momentum * c_inc + (epsilonc / N) * (Y0.rowwise().sum()- Y1.rowwise().sum());



	W = W + W_inc;
	b = b + b_inc;
	c = c + c_inc;


	return err;
}

//NumericType restricted_boltzmann_machine::train(const shared_ptr<dataset> & X, int num_iter)
//{
//	if (num_iter == 0)
//		return numeric_limits<NumericType>::quiet_NaN();
//
//	// change from initial to final momentum is a "compile time" parameter
//	int MOMENTUM_THRESHOLD = ceil(float(num_iter) / 4);
//
//	int N = X->get_sample_num();
//
//	// initialize weights and biases
//	W = (0.1*randn<MatrixType>(numhid,numvis));
//
//	b.setZero(numvis);
//	c.setZero(numhid);
//
//	// initialize variables for 1 step constrastive divergence
//	W_inc.setZero(numhid,numvis);
//	b_inc.setZero(numvis);
//	c_inc.setZero(numhid);
//
//	cur_momentum = initialmomentum;
//
//
//	MatrixType samples = X->get_data();
//
//	train_error.resize(num_iter);
//
//	// do contrastive divergence
//	for (int curr_iter = 0; curr_iter < num_iter; curr_iter++)
//	{
//
//		NumericType tot_err = 0;
//
//		// update momentum
//		if (curr_iter > MOMENTUM_THRESHOLD)
//			cur_momentum = finalmomentum;
//
//
//		tot_err += train_one_step( samples);
//
//		train_error[curr_iter] = tot_err;
//	}
//
//	return train_error[num_iter-1];
//}


NumericType restricted_boltzmann_machine::train(const shared_ptr<dataset> & X, int num_iter)
{
	if (num_iter == 0)
		return numeric_limits<NumericType>::quiet_NaN();

	// change from initial to final momentum is a "compile time" parameter
	int MOMENTUM_THRESHOLD = ceil(float(num_iter) / 4);

	int N = X->get_sample_num();

	// initialize weights and biases
	W = (0.1*randn<MatrixType>(numhid,numvis));

	b.setZero(numvis);
	c.setZero(numhid);

	// initialize variables for 1 step constrastive divergence
	W_inc.setZero(numhid,numvis);
	b_inc.setZero(numvis);
	c_inc.setZero(numhid);

	cur_momentum = initialmomentum;


	MatrixType samples = X->get_data();

	train_error.resize(num_iter);


	// do contrastive divergence
	for (int curr_iter = 0; curr_iter < num_iter; curr_iter++)
	{

		NumericType tot_err = 0;

		// update momentum
		if (curr_iter > MOMENTUM_THRESHOLD)
			cur_momentum = finalmomentum;

		if (batch_size == 0)
		{
			for (int i = 0; i < iter_per_batch;i++)
				tot_err += train_one_step( samples);
		}
		else
		{

			int batch_num = (N + batch_size -1)/batch_size;

			if (batch_num == 1)
			{
				for (int i = 0; i < iter_per_batch;i++)
					tot_err += train_one_step( samples);
			}
			else
			{

				random_shuffer_dataset_splitter splitter(batch_num);
				dataset_group group = splitter.split(*X);

				for (int batch_id = 0; batch_id < group.get_dataset_num(); batch_id++)
				{
					shared_ptr<dataset> cur_data = group.get_dataset(batch_id);
					MatrixType cur_samples = cur_data->get_data();

					for (int i = 0; i < iter_per_batch;i++)
						tot_err += train_one_step( cur_samples);

				}

				tot_err /= group.get_dataset_num();
			}
		}

		train_error[curr_iter] = tot_err;

//		std::cout << "RBM Learning iter : " << curr_iter << " Finished !"<<std::endl;

	}

	return train_error[num_iter-1];
}

MatrixType restricted_boltzmann_machine::output(const MatrixType & data)
{
	MatrixType result;
	if (type == linear)
	{
		//do linear level
		linear_layer_output(result, W, 'N', data, c);
		// matrix_linear_transform(Y.data(), numhid, numvis, Y.cols(), 1.0, W.data(), 'N', 	data.data(), 1, c.data());

	}
	else
	{
		// do logistic unit levels
		logistic_layer_output(result, W, 'N', data, c);
		// matrix_logistic_transform(Y.data(), numhid, numvis, Y.cols(), -1.0, W.data(), 'N', data.data(), -1, c.data());
	}

	return result;
}

shared_ptr<dataset> restricted_boltzmann_machine::output(const dataset & X)
{
	MatrixType cur_Y = output(X.get_data());
	return X.clone_update_data(cur_Y);
}

shared_ptr<dataset> restricted_boltzmann_machine::output_batch(const dataset & X,int batch_size)
{

	int N = X.get_sample_num();
	int batch_num = (N + batch_size -1)/batch_size;

	MatrixType Y(numhid,N);
	for (int i = 0;i<batch_num;i++)
	{
		int cur_batch_size = batch_size;
		if (i == batch_num-1)
			cur_batch_size = N - (batch_num-1)*batch_size;

		MatrixType cur_X = X.get_data().block(0,i*batch_size,X.get_dim(),cur_batch_size);
		Y.block(0,i*batch_size,numhid,cur_batch_size) = output(cur_X);
		
	}

	return X.clone_update_data(Y);
}


MatrixType output_batch(const MatrixType & data, int batch_size);

MatrixType & restricted_boltzmann_machine::get_W()
{
	return W;
}
VectorType & restricted_boltzmann_machine::get_b()
{
	return b;
}
VectorType & restricted_boltzmann_machine::get_c()
{
	return c;
}
