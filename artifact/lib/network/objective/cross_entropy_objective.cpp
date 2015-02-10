/*
* cross_entropy_objective.cpp
*
*  Created on: 2010-6-18
*      Author: sun
*/

#include <liblearning/deep/objective/cross_entropy_objective.h>

#include <liblearning/util/matrix_util.h>


#include <liblearning/deep/deep_auto_encoder.h>
#include <liblearning/deep/neuron_layer_operation.h>
using namespace deep;
using namespace deep::objective;


cross_entropy_objective::cross_entropy_objective()
{
	type = decoder_related;

}

cross_entropy_objective::~cross_entropy_objective()
{
}



NumericType cross_entropy_objective::prepared_value(deep_auto_encoder & net) 
{
//	const MatrixType & R = net.get_layered_output(net.get_output_layer_id());

	const MatrixType & act = net.get_last_layer_activation();
	
	NumericType N = data_set->get_sample_num();
	
	const MatrixType & X = data_set->get_data();

#if defined(USE_GPU)
	NumericType err = gpumatrix::cross_entropy(X, act);
#elif  defined(USE_PARTIAL_GPU)
	GPUMatrixType gX = X;
	GPUMatrixType gAct = act;
	NumericType err = gpumatrix::cross_entropy(gX, gAct);
#else
	NumericType err = 1.0/N * ((1+ act.array().exp()).log() - act.array()*X.array()).sum();
//	NumericType err = -1.0/N*( X.array()*R.array().log() + (1-X.array())*(1-R.array()).log() ).sum();
#endif
	current_objective_val = err;
	return err;

}
vector<shared_ptr<MatrixType>> cross_entropy_objective::prepared_value_delta(deep_auto_encoder & net) 
{
	const vector<neuron_type> & neuron_types = net.get_neuron_types();

	if (neuron_types[net.get_output_layer_id()] == linear)
		throw runtime_error("cross entropy objective does not fit for linear neurons");
	
	const MatrixType & act = net.get_last_layer_activation();
	

	const MatrixType & R = net.get_layered_output(net.get_output_layer_id());
	
	const MatrixType & X = data_set->get_data();

	NumericType N = data_set->get_sample_num();

#if defined(USE_GPU)
	MatrixType delta_mat = gpumatrix::cross_entropy_delta(X, R);
#elif  defined(USE_PARTIAL_GPU)
	GPUMatrixType gX = X;
	GPUMatrixType gR = R;
	MatrixType delta_mat = gpumatrix::cross_entropy_delta(gX, gR);
#else
	
	MatrixType delta_mat = 1.0/N*( (1-X.array())*R.array() - X.array()*(1-R.array()));
#endif


	shared_ptr<MatrixType> error_diff( new MatrixType( delta_mat));

	vector<shared_ptr<MatrixType>> result(2);

	result[0] = shared_ptr<MatrixType>();
	result[1] = error_diff;

	return result;
}


cross_entropy_objective * cross_entropy_objective::clone()
{
	return new cross_entropy_objective(*this);
}

string cross_entropy_objective::get_info()
{
	return "CrossEntropy Objective Value = " + boost::lexical_cast<string>(current_objective_val);
}
