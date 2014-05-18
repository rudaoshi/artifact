#include <liblearning/deep/objective/data_related_network_objective.h>

#include <liblearning/deep/deep_auto_encoder.h>

using namespace deep;
using namespace deep::objective;

data_related_network_objective::data_related_network_objective()
{
}


data_related_network_objective::~data_related_network_objective(void)
{
}

void data_related_network_objective::set_dataset(const shared_ptr<dataset> & data_set_)
{
	data_set = data_set_;
}




tuple<NumericType, VectorType> data_related_network_objective::value_diff(deep_auto_encoder & net) 
{
	if(type == encoder_related)
	{
//		std::cout << "Begin Computing Encoder_Related Objective" << std::endl;
		net.encode(data_set->get_data());

		NumericType value =  prepared_value(net);
		vector<shared_ptr<MatrixType>> error_diff = prepared_value_delta(net);

		//MatrixType encoder_delta = net.error_diff_to_delta(*error_diff[0],net.get_coder_layer_id());
		net.zero_dWb();
		net.backprop_encoder_to_input(*error_diff[0]);

//		std::cout << "Finishe Computing Encoder_Related Objective" << std::endl;
		return make_tuple(value,net.get_dWb());
	}
	else if(type == decoder_related)
	{
//		std::cout << "Begin Computing Decoder_Related Objective" << std::endl;

		MatrixType feature = net.encode(data_set->get_data());
		net.decode(feature);

//		std::cout << "Computing Feature and Reconstruction Objective" << std::endl;

//		std::cout << "Computing prepared_value" << std::endl;
		NumericType value =  prepared_value(net);
//		std::cout << "Computing prepared_value_diff" << std::endl;
		vector<shared_ptr<MatrixType>> error_diff = prepared_value_delta(net);

//		std::cout << "Computing diff[1] error_diff_to_delta" << std::endl;
		//MatrixType output_delta = net.error_diff_to_delta(*error_diff[1],net.get_output_layer_id());

		MatrixType output_delta = *error_diff[1];

//		std::cout << "Computing backprop_output_to_encoder" << std::endl;
		net.backprop_output_to_encoder(output_delta);

		if (error_diff[0])
		{
//			std::cout << "Computing diff[0] error_diff_to_delta" << std::endl;
			//MatrixType encoder_delta = net.error_diff_to_delta(*error_diff[0],net.get_coder_layer_id());
//			std::cout << "Adding Decoder Related with Encoder Related"<< std::endl;
			output_delta += *error_diff[0];
		}

//		std::cout << "Computing backprop_encoder_to_input" << std::endl;
		net.backprop_encoder_to_input(output_delta);

//		std::cout << "Finish Computing Decoder_Related Objective" << std::endl;

		return make_tuple(value,net.get_dWb());
	}
}

NumericType data_related_network_objective::value(deep_auto_encoder & net) 
{

	if(type == encoder_related)
	{
		net.encode(data_set->get_data());
		return prepared_value(net);
	}
	else if(type == decoder_related)
	{
		MatrixType feature = net.encode(data_set->get_data());
		net.decode(feature);
		return prepared_value(net);
	}

}