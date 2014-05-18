#ifndef RANDOM_LAYERWISE_INITIALIZE_H
#define RANDOM_LAYERWISE_INITIALIZE_H


#include <liblearning/deep/layerwise_initializer.h>
#include <liblearning/deep/deep_auto_encoder.h>

#include <liblearning/core/dataset.h>


#include <liblearning/deep/objective/data_related_network_objective.h>

namespace deep
{
	namespace initializer
	{
		using namespace deep;
class POCO_EXPORT random_layerwise_initializer :public layerwise_initializer
{

	
	shared_ptr<deep_auto_encoder> aem;
	shared_ptr<dataset> current_dataset;


public:
	
	random_layerwise_initializer();

	random_layerwise_initializer(const random_layerwise_initializer & ae_init);

	virtual ~random_layerwise_initializer(void);

	virtual void init(int input_dim, int output_dim, neuron_type type_);
	virtual NumericType train(const shared_ptr<dataset> & train_data);

	
	virtual shared_ptr<dataset> get_output();

	virtual MatrixType get_W1();
	virtual VectorType get_b1();

	virtual MatrixType get_W2();
	virtual VectorType get_b2();


	virtual random_layerwise_initializer * clone();


};
	}
}
#endif