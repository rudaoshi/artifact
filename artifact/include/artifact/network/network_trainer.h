
#ifndef ARTIFACT_NETWORK_NETWORK_CREATEOR_H_
#define ARTIFACT_NETWORK_NETWORK_CREATEOR_H_

#include <artifact/config.h>


namespace artifact
{
    namespace network
    {
        struct training_param
        {

        };

        struct training_context
        {

        };

        struct sgd_training_param: public training_param
        {
            NumericType learning_rate;
            NumericType decay_rate;
            int batch_size;
            int max_epoches;
        };

        struct gd_training_param: public training_param
        {
            NumericType learning_rate;
            NumericType decay_rate;
            int max_epoches;
        };

        class network_trainer
        {
            deep_network train(deep_network & net,
                    const MatrixType & X, const MatrixType & y,
                    const training_param & param,
                    const training_context * context = 0) = 0;

        };

        class gd_network_trainer: public network_trainer
        {
            deep_network train(deep_network & net,
                    const MatrixType & X, const MatrixType & y,
                    const training_param & param,
                    const training_context * context = 0);
        };

        class sgd_network_trainer: public network_trainer
        {
            deep_network train(deep_network & net,
                    const MatrixType & X, const MatrixType & y,
                    const training_param & param,
                    const training_context * context = 0);
        };
    }
}

#endif