/*
 * network_auto_encoder.h
 *
 *  Created on: 2010-6-3
 *      Author: sun
 */

#ifndef ARTIFACT_NETWORK_DEEPNETWORK_H_
#define ARTIFACT_NETWORK_DEEPNETWORK_H_

#include <vector>
#include <memory>

using namespace std;

#include <artifact/config.h>
#include <artifact/network/layer/mlp_layer.h>

#include <artifact/machine/machine.h>
#include <artifact/optimization/optimizable.h>

using namespace artifact::machines;
using namespace artifact::optimization;

namespace artifact{
    namespace network {


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

            void add_layer(const mlp_layer &layer);

            void remove_layer(int pos);

            mlp_layer & get_layer(int pos);

        public:


            int get_layer_num();


        public:

            //		network_auto_encoder(const vector<int>& structure,  const vector<neuron_type>& neuron_type);

            deep_network();

            virtual ~deep_network();

            MatrixType predict(const MatrixType & X);

            VectorType predict(const VectorType & x);

            virtual VectorType get_parameter();
            virtual void set_parameter(const VectorType &parameter_);

            virtual NumericType objective(const MatrixType & x,
                    const VectorType & y);
            /**
            * partial output/partial param
            */
            virtual pair<NumericType, VectorType> gradient(const MatrixType & x,
                    const VectorType & y);


        };
    }
}

#endif /* BP_NETWORK_H_ */
