/*
* mse_decoder_objective.cpp
*
*  Created on: 2010-6-18
*      Author: sun
*/

#include <liblearning/deep/objective/mse_objective.h>

#include <liblearning/util/matrix_util.h>


#include <liblearning/deep/deep_auto_encoder.h>
#include <liblearning/deep/neuron_layer_operation.h>
using namespace deep;
using namespace deep::objective;
mse_objective::mse_objective()
{
	type = decoder_related;

}

mse_objective::~mse_objective()
{
}

NumericType mse_objective::prepared_value(deep_auto_encoder & net) 
{
	const MatrixType & reconstruction = net.get_layered_output(net.get_output_layer_id());

	NumericType N = data_set->get_sample_num();

	NumericType err = (reconstruction-data_set->get_data()).squaredNorm();
	NumericType mse = 1/N*err;

	cur_obj_val = mse;

	return mse;

}
vector<shared_ptr<MatrixType>> mse_objective::prepared_value_delta(deep_auto_encoder & net) 
{
	const MatrixType & reconstruction = net.get_layered_output(net.get_output_layer_id());

	NumericType N = data_set->get_sample_num();

	shared_ptr<MatrixType> error_diff( new MatrixType( 2.0/N * (reconstruction - data_set->get_data())));


	if (net.get_neuron_type_of_layer(net.get_output_layer_id()) == logistic)
	{

		*error_diff = net.error_diff_to_delta(*error_diff, net.get_output_layer_id());
	}

	vector<shared_ptr<MatrixType>> result(2);

	result[0] = shared_ptr<MatrixType>();
	result[1] = error_diff;

	return result;
}


mse_objective * mse_objective::clone()
{
	return new mse_objective(*this);
}

string  mse_objective::get_info()
{
	return "MSE Objective Value = " + boost::lexical_cast<string>(cur_obj_val);
}