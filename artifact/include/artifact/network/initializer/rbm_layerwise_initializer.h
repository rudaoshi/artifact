#ifndef RBM_LAYERWISE_INITIALIZE_H
#define RBM_LAYERWISE_INITIALIZE_H

#include <liblearning/deep/layerwise_initializer.h>
#include <liblearning/deep/restricted_boltzmann_machine.h>

#include <liblearning/core/dataset.h>
#include <liblearning/core/clonable_object.h>

namespace deep
{
	namespace initializer
	{
				using namespace deep;
class rbm_layerwise_initializer :public layerwise_initializer
{

	
	shared_ptr<restricted_boltzmann_machine> rbm;
	shared_ptr<dataset> current_dataset;

	int max_iter;

	int batch_size;

	int iter_per_batch;

public:


	
	rbm_layerwise_initializer(int max_iter,int batch_size = 0, int iter_per_batch = 0);
	virtual ~rbm_layerwise_initializer(void);

	virtual void init(int input_dim, int output_dim, neuron_type type_);
	virtual NumericType train(const shared_ptr<dataset> & train_data);
	virtual shared_ptr<dataset> get_output();

	virtual MatrixType get_W1();
	virtual VectorType get_b1();

	virtual MatrixType get_W2();
	virtual VectorType get_b2();

	virtual rbm_layerwise_initializer * clone();

};

	}
}
#endif