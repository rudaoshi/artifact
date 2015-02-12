                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    #include <liblearning/deep/initializer/ae_layerwise_initializer.h>

using namespace deep::initializer;

ae_layerwise_initializer::ae_layerwise_initializer(int rbm_iter_,  int batch_size_, int iter_per_batch_, int batch_finetune_iter_,
	const shared_ptr<network_objective> & object_, 
	const shared_ptr<optimizer> & finetune_optimizer_
	)
	:object(object_),finetune_optimizer(finetune_optimizer_),
	rbm_iter(rbm_iter_),batch_size(batch_size_),iter_per_batch(iter_per_batch_),batch_finetune_iter(batch_finetune_iter_)
{
}

ae_layerwise_initializer::ae_layerwise_initializer(const ae_layerwise_initializer & ae_init)
	:rbm_iter(ae_init.rbm_iter),batch_size(ae_init.batch_size),iter_per_batch(ae_init.iter_per_batch),batch_finetune_iter(ae_init.batch_finetune_iter),
	object(ae_init.object->clone()),finetune_optimizer(ae_init.finetune_optimizer->clone())
{
}

ae_layerwise_initializer::~ae_layerwise_initializer(void)
{
}

void ae_layerwise_initializer::init(int input_dim, int output_dim, neuron_type type_)
{
	std::vector<int> cur_structure(2);
	std::vector<neuron_type> cur_type(1);

	cur_structure[0] = input_dim;
	cur_structure[1] = output_dim;
	cur_type[0] = type_;

	aem.reset(new deep_auto_encoder(cur_structure, cur_type));

	aem->set_batch_setting(batch_size, iter_per_batch);


}

NumericType ae_layerwise_initializer::train(const shared_ptr<dataset> & train_data)
{
	current_dataset =  train_data;
	aem->init_stacked_rbm(*train_data, rbm_iter);

	aem->set_optimizer(finetune_optimizer);

//	if (batch_size == 0)
		return aem->finetune(train_data, *object);
	//else
	//	return aem->finetune_batch(train_data,*object, batch_size, batch_finetune_iter);

}

MatrixType ae_layerwise_initializer::get_W1()
{
	return aem->get_W(0);
}

shared_ptr<dataset> ae_layerwise_initializer::get_output()
{
//	if (batch_size == 0)
		return aem->encode(*current_dataset);
	//else
	//	return aem->encode_batch(*current_dataset,batch_size);
}

VectorType ae_layerwise_initializer::get_b1()
{
	return aem->get_b(0);
}

MatrixType ae_layerwise_initializer::get_W2()
{
	return aem->get_W(1);
}

VectorType ae_layerwise_initializer::get_b2()
{
	return aem->get_b(1);
}

ae_layerwise_initializer * ae_layerwise_initializer::clone()
{
	return new ae_layerwise_initializer(*this);
}