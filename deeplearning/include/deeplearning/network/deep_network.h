/*
 * network_auto_encoder.h
 *
 *  Created on: 2010-6-3
 *      Author: sun
 */

#ifndef BP_NETWORK_H_
#define BP_NETWORK_H_

#include <deeplearning/core/dataset.h>
#include <deeplearning/core/serialize.h>
#include <deeplearning/network/neuron_type.h>

#include <deeplearning/network/layerwise_initializer.h>
#include <deeplearning/network/network_objective.h>

#include <deeplearning/core/maker.h>
#include <deeplearning/core/feature_extractor.h>

#include <deeplearning/optimization/optimizer.h>
#include <deeplearning/core/evaluator.h>

#include <deeplearning/machine/machine.h>
#include <deeplearning/optimization/optimizable.h>


#include <vector>
#include <memory>
using namespace std;



namespace network
{

	struct network_network_param
	{
		vector<int> structure;
		vector<neuron_type> neuron_types;

		shared_ptr<layerwise_initializer> initializer;

		shared_ptr<network_objective> objective;

		shared_ptr<optimization::optimizer> finetune_optimizer;

		shared_ptr<evaluator<NumericType> > perf_evaluator;

		int batch_size;

		int iter_per_batch;

		int finetune_iter_num;

		int code_layer_id;

	};

//	using Eigen::Map;

	class deep_network: public machine<VectorType, VectorType>,
                    public gradient_optimizable<MatrixType, VectorType>
                    public parameterized<VectorType>
	{


	protected:

// layers
		vector<bp_layer > layers;

    void feed_forward(const MatrixType & X);
    void back_propagate();

	public:

	    void add_layer(const bp_layer & layer);
      void remove_layer(int pos);

	    bp_layer & get_layer(int pos);

	public:


		int get_layer_num();

		const MatrixType & get_layered_input(int id);

		const MatrixType & get_layered_output(int id);

	public:

//		network_auto_encoder(const vector<int>& structure,  const vector<neuron_type>& neuron_type);

		bp_network(const bp_network & net_);
		bp_network();
		virtual ~bp_network();


#pragma region Implementing Interface For machine

		virtual VectorType predict(const VectorType & data) = 0;
    virtual MatrixType predict(const MatrixType & data) = 0;


#pragma endregion


    virtual NumicalType objective(const InputDataSetType & traindata) = 0;

    virtual ParameterType gradient(const InputDataSetType & testdata) = 0;


	};
}

#endif /* BP_NETWORK_H_ */
