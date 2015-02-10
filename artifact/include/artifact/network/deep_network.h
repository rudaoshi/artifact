/*
 * network_auto_encoder.h
 *
 *  Created on: 2010-6-3
 *      Author: sun
 */

#ifndef BP_NETWORK_H_
#define BP_NETWORK_H_

#include <artifact/config.h>
#include <artifact/network/layer/mlp_layer.h>

#include <artifact/core/machine/machine.h>
#include <artifact/core/optimization/optimizable.h>


#include <vector>
#include <memory>

using namespace std;

namespace artifact{
    namespace network {

        struct network_network_param {
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


        struct batch_layer_output {
            NumericType cost;
            MatrixType output;
        };


        //	using Eigen::Map;

        class deep_network : public machine<VectorType, VectorType>,
                             public gradient_optimizable<MatrixType, VectorType>
                             public parameterized<VectorType> {


        protected:

            // layers
            vector<backpropagation_layer> layers;

            vector<batch_layer_output> feed_forward(const MatrixType &input);

            vector<VectorType> back_propagate(const MatrixType &input, const vector<batch_layer_output> &laywise_output);

        public:

            void add_layer(const bp_layer &layer);

            void remove_layer(int pos);

            backpropagation_layer &get_layer(int pos);

        public:


            int get_layer_num();


        public:

            //		network_auto_encoder(const vector<int>& structure,  const vector<neuron_type>& neuron_type);

            deep_network(const deep_network &net_);

            deep_network();

            virtual ~deep_network();


            virtual NumericType cost(const MatrixType &traindata) = 0;

            virtual VectorType param_gradient(const MatrixType &traindata) = 0;

            virtual VectorType cost_gradient(const MatrixType &traindata) = 0;


        };
    }
}

#endif /* BP_NETWORK_H_ */
