#include <liblearning/deep/initializer/rbm_layerwise_initializer.h>

using namespace deep::initializer;
rbm_layerwise_initializer::rbm_layerwise_initializer(int max_iter_,int batch_size_, int iter_per_batch_)
	:max_iter(max_iter_),batch_size(batch_size_),iter_per_batch(iter_per_batch_)
{
}


rbm_layerwise_initializer::~rbm_layerwise_initializer(void)
{
}

void rbm_layerwise_initializer::init(int input_dim, int output_dim, neuron_type type_)
{
	rbm.reset(new restricted_boltzmann_machine(input_dim,output_dim,type_));
	rbm->set_batch_setting(batch_size, iter_per_batch);
}

NumericType rbm_layerwise_initializer::train(const shared_ptr<dataset> & train_data)
{
	current_dataset = train_data;

//	if (batch_size == 0)
		return rbm->train(train_data,max_iter);
	//else
	//	return rbm->train_batch(train_data,batch_size,iter_per_batch, max_iter);
}

shared_ptr<dataset> rbm_layerwise_initializer::get_output()
{
//	if (batch_size == 0)
		return rbm->output(*current_dataset);
	//else
	//	return rbm->output_batch(*current_dataset,batch_size);
}

MatrixType rbm_layerwise_initializer::get_W1()
{
	return rbm->get_W();
}

VectorType rbm_layerwise_initializer::get_b1()
{
	return rbm->get_c();
}

MatrixType rbm_layerwise_initializer::get_W2()
{
	return rbm->get_W().transpose();
}

VectorType rbm_layerwise_initializer::get_b2()
{
	return rbm->get_b();
}


rbm_layerwise_initializer * rbm_layerwise_initializer::clone()
{
	return new rbm_layerwise_initializer(*this);
}