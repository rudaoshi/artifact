#ifndef ARTIFACT_NETWORK_MLPLAYER_H
#define ARTIFACT_NETWORK_MLPLAYER_H



#include <artifact/config.h>
#include <artifact/machine/machine.h>
#include <artifact/loss/loss.h>
#include <artifact/network/layer/activator.h>


#include <memory>
#include <algorithm>
using namespace std;

using namespace artifact::losses;
using namespace artifact::machines;

namespace artifact
{
    namespace network {

        class mlp_layer : public machine
        {

        public:

            int input_dim;
            int output_dim;

            MatrixType W;
            VectorType b;

            std::shared_ptr<loss_function> loss_func;

            std::shared_ptr<activator> active_func;

        public:

            bool is_loss_contributor() const;
            int get_input_dim() const;
            int get_output_dim() const;


        public:

            /**
            * Following are group of methods used in back-prop algorithm
            * Because the back-prop of delta involves two layers, it is difficult to write it gracefully.
            * Insted, we back-prop loss-gradient from one layer, and then compute delta in the former layer
            */
            MatrixType compute_loss_gradient(const MatrixType & output, const VectorType & y);

            MatrixType backprop_loss_gradient(const MatrixType & delta);

            MatrixType compute_delta(const MatrixType & activator,
                    const MatrixType & output,
                    const MatrixType & loss_gradient);

            pair<MatrixType, VectorType> compute_param_gradient(const MatrixType & input, const MatrixType & delta);



//            MatrixType compute_delta(const MatrixType & activator,
//                    const MatrixType & output,
//                    const VectorType & y);



//            MatrixType backprop_delta(const MatrixType & delta,
//                    const MatrixType & former_activator,
//                    const MatrixType & input
//                    );

            pair<MatrixType, MatrixType> predict_with_activator(const MatrixType & X);

        public:
            mlp_layer(int input_dim, int output_dim, shared_ptr<activator> active_func);

            virtual void set_loss(const shared_ptr<loss_function> & loss_func_ );

            virtual VectorType predict(const VectorType &testdata);

            virtual MatrixType
                    predict(const MatrixType &test_set);



        };
    }
}

#endif
