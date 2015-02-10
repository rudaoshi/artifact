#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H


#include <artifact/config.h>
#include <artifact/network/layer/mlp_layer.h>

namespace artifact
{
    namespace network {
        class linear_layer : public mlp_layer {
        public:

            linear_layer(int input_dim, int output_dim);

            virtual MatrixType predict(const MatrixType &input);

            virtual VectorType predict(const VectorType &input);


            virtual MatrixType compute_delta(const MatrixType &input, const MatrixType &output);

            virtual MatrixType backprop_delta(const MatrixType &delta, const MatrixType &input, const MatrixType &output);


        };
    }
}

#endif
