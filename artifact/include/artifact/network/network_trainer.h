
#ifndef ARTIFACT_NETWORK_NETWORK_TRAINER_H_
#define ARTIFACT_NETWORK_NETWORK_TRAINER_H_

#include <artifact/config.h>
#include <artifact/network/deep_network.h>
#include <artifact/optimization/optimizer.h>

using namespace artifact::optimization;

namespace artifact
{
    namespace network
    {

        class network_trainer
        {

        public:
            virtual deep_network train(deep_network & net,
                    const MatrixType & X, const MatrixType * y = 0) = 0;

        };

        class optimization_trainer: public network_trainer
        {

            optimizer & optimizer_;

        public:

            optimization_trainer(optimizer & optimizer__);

            virtual deep_network train(deep_network & net,
                    const MatrixType & X, const MatrixType * y = 0);
        };

    }
}

#endif