#include <liblearning/deep/initializer/random_layerwise_initializer.h>

using namespace deep::initializer;
random_layerwise_initializer::random_layerwise_initializer()
{
}

random_layerwise_initializer::random_layerwise_initializer(const random_layerwise_initializer & ae_init)
{
}

random_layerwise_initializer::~random_layerwise_initializer(void)
{
}

void random_layerwise_initializer::init(int input_dim, int output_dim, neuron_type type_)
{
	std::vector<int> cur_structure(2);
	std::vector<neuron_type> cur_type(1);    

	cur_structure[0] = input_dim;
	cur_structure[1] = output_dim;
	cur_type[0] = type_;

	aem.reset(new deep_auto_encoder(cur_structure, cur_type));
}

NumericType random_layerwise_initializer::train(const shared_ptr<dataset> & train_data)
{
	current_dataset =  train_data;
	aem->init_random();

	return 0;

}

MatrixType random_layerwise_initializer::get_W1()
{
	return aem->get_W(0);
}

shared_ptr<dataset> random_layerwise_initializer::get_output()
{
	return aem->encode(*current_dataset);
}

VectorType random_layerwise_initializer::get_b1()
{
	return aem->get_b(0);
}

MatrixType random_layerwise_initializer::get_W2()
{
	return aem->get_W(1);
}

VectorType random_layerwise_initializer::get_b2()
{
	return aem->get_b(1);
}

random_layerwise_initializer * random_layerwise_initializer::clone()
{
	return new random_layerwise_initializer(*this);
}