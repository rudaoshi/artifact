
#ifndef ARTIFACT_NETWORK_NETWORK_TRAINER_H_
#define ARTIFACT_NETWORK_NETWORK_TRAINER_H_

#include <artifact/config.h>
#include <artifact/network/deep_network.h>

namespace artifact
{
    namespace network
    {
        struct training_setting
        {

        };

        struct training_context
        {

        };

        struct sgd_training_setting: public training_setting
        {
            NumericType learning_rate;
            NumericType decay_rate;
            int batch_size;
            int max_epoches;
        };

        struct gd_training_setting: public training_setting
        {
            NumericType learning_rate;
            NumericType decay_rate;
            int max_epoches;
        };

        class network_trainer
        {
        public:
            virtual deep_network train(deep_network & net,
                    const training_setting & param,
                    const MatrixType & X, const VectorType * y = 0,
                    const training_context * context = 0) = 0;

        };

        class gd_network_trainer: public network_trainer
        {
        public:
            virtual deep_network train(deep_network & net,
                    const training_setting & param,
                    const MatrixType & X, const VectorType * y = 0,
                    const training_context * context = 0);
        };

        class sgd_network_trainer: public network_trainer
        {
        public:
            virtual deep_network train(deep_network & net,
                    const training_setting & param,
                    const MatrixType & X, const VectorType * y = 0,
                    const training_context * context = 0);
        };
    }
}

#endif