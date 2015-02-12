
#include <artifact/network/network_trainer.h>

using namespace artifact::network;


deep_network sgd_network_trainer::train(deep_network & net,
        const MatrixType & X, const MatrixType & y,
        const training_param & param,
        const training_context * context = 0)
{
    const sgd_training_param & param = dynamic_cast<const sgd_training_param &>(param);




}
