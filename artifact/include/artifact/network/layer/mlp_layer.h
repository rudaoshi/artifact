#ifndef BACKPROP_NEURON_LAYER_H
#define BACKPROP_NEURON_LAYER_H

#include <memory>

using namespace std;

#include <artifact/config.h>
#include <artifact/core/machine/machine.h>
#include <artifact/core/loss/loss.h>
#include <artifact/core/optimization/optimizable.h>

using namespace artifact::core;
namespace artifact
{
    namespace network {

        class mlp_layer : public machine
        {
        protected:

            int input_dim;
            int output_dim;

            MatrixType W;
            VectorType b;

            shared_ptr<loss> loss_func;

        public:


            int get_input_dim();

            int get_output_dim();

        protected:

            virtual VectorType compute_param_gradient(const MatrixType &delta, const MatrixType &input, const MatrixType &output);

            virtual MatrixType compute_delta(const MatrixType &input, const MatrixType &output) = 0;

            virtual MatrixType backprop_delta(const MatrixType &delta, const MatrixType &input, const MatrixType &output) = 0;


        public:
            mlp_layer(int input_dim, int output_dim);

            virtual const VectorType &get_parameter();

            virtual void set_parameter(const VectorType &parameter_);

            virtual void set_object(const shared_ptr<objective> & object );

        };
    }
}

#endif
