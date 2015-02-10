#ifndef BACKPROP_NEURON_LAYER_H
#define BACKPROP_NEURON_LAYER_H

#include <

#include <artifact/config.h>
#include <artifact/core/machine/machine.h>
#include <artifact/core/objective/objective.h>

using namespace artifact::core;
namespace artifact
{
    namespace network {

        class mlp_layer : public machine,
                          public parameterized<VectorType>
        {
        protected:

            int input_dim;
            int output_dim;

            MatrixType W;
            VectorType b;

            shared_ptr<objective> object;

        public:


            int get_input_dim();

            int get_output_dim();


        public:
            mlp_layer(int input_dim, int output_dim);

            virtual VectorType compute_param_gradient(const MatrixType &delta, const MatrixType &input, const MatrixType &output);

            virtual MatrixType compute_delta(const MatrixType &input, const MatrixType &output) = 0;

            virtual MatrixType backprop_delta(const MatrixType &delta, const MatrixType &input, const MatrixType &output) = 0;

            virtual const VectorType &get_parameter();

            virtual void set_parameter(const VectorType &parameter_);

            virtual void set_object(const shared_ptr<objective> & object );



        };
    }
}

#endif
