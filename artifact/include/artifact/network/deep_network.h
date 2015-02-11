/*
 * network_auto_encoder.h
 *
 *  Created on: 2010-6-3
 *      Author: sun
 */

#ifndef ARTIFACT_NETWORK_DEEPNETWORK_H_
#define ARTIFACT_NETWORK_DEEPNETWORK_H_

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

        class deep_network : public machine,
                             public gradient_optimizable
        {


        protected:

            // layers
            vector<mlp_layer> layers;

            vector<pair<MatrixType, MatrixType>> feed_forward(const MatrixType &input);

            vector<pair<MatrixType, VectorType>> back_propagate(const MatrixType &input,
                    const VectorType & y,
                    const vector<pair<MatrixType, MatrixType>> &laywise_output);

        public:

            void add_layer(const bp_layer &layer);

            void remove_layer(int pos);

            mlp_layer & get_layer(int pos);

        public:


            int get_layer_num();


        public:

            //		network_auto_encoder(const vector<int>& structure,  const vector<neuron_type>& neuron_type);

            deep_network(const deep_network &net_);

            deep_network();

            virtual ~deep_network();


            virtual VectorType get_parameter();
            virtual void set_parameter(const VectorType &parameter_);

            virtual NumericType objective(const MatrixType & x,
                    const VectorType & y) = 0;
            /**
            * partial output/partial param
            */
            virtual VectorType gradient(const MatrixType & x,
                    const VectorType & y) = 0;


        };
    }
}

#endif /* BP_NETWORK_H_ */
