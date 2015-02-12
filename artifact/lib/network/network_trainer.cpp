
#include <artifact/network/network_trainer.h>
#include <artifact/optimization/gd_optimizer.h>

using namespace artifact::network;


deep_network gd_network_trainer::train(deep_network & net,
        const MatrixType & X, const MatrixType & y,
        const training_param & param,
        const training_context * context)
{
    const gd_training_param & param = dynamic_cast<const gd_training_param &>(param);

    gd_optimizer optimizer(param.learning_rate, param.decay_rate, param.max_epoches);

    VectorType param = net.get_parameter();
    NumericType fval = 0;
    tie(fval, param) = optimizer.optimize(net, param, X, y);

    net.set_parameter(param);

    return net;

}
