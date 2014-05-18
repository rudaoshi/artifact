#ifndef AE_LAYERWISE_INITIALIZE_H
#define AE_LAYERWISE_INITIALIZE_H


#include <liblearning/deep/layerwise_initializer.h>
#include <liblearning/deep/deep_auto_encoder.h>

#include <liblearning/core/dataset.h>


#include <liblearning/deep/network_objective.h>

#include <liblearning/optimization/optimizer.h>

namespace deep
{
	namespace initializer
	{
		using namespace deep;
		using namespace optimization;
		class POCO_EXPORT ae_layerwise_initializer :public layerwise_initializer
		{


			shared_ptr<deep_auto_encoder> aem;
			shared_ptr<dataset> current_dataset;

			shared_ptr<network_objective> object;

			shared_ptr<optimizer> finetune_optimizer;

			int rbm_iter;

			int batch_size;
			
			int iter_per_batch;

			int batch_finetune_iter;

		public:

			ae_layerwise_initializer(int rbm_iter, int batch_size, int iter_per_batch, int batch_finetune_iter, const shared_ptr<network_objective> & object, const shared_ptr<optimizer> & finetune_optimizer_);

			ae_layerwise_initializer(const ae_layerwise_initializer & ae_init);

			virtual ~ae_layerwise_initializer(void);

			virtual void init(int input_dim, int output_dim, neuron_type type_);
			virtual NumericType train(const shared_ptr<dataset> & train_data);


			virtual shared_ptr<dataset> get_output();

			virtual MatrixType get_W1();
			virtual VectorType get_b1();

			virtual MatrixType get_W2();
			virtual VectorType get_b2();


			virtual ae_layerwise_initializer * clone();


		};
	}
}
#endif