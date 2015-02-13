
#include <artifact/network/network_trainer.h>
#include <artifact/optimization/gd_optimizer.h>

using namespace artifact::network;


deep_network gd_network_trainer::train(deep_network & net,
        const training_setting & setting,
        const MatrixType & X, const VectorType * y,
        const training_context * context)
{
    const gd_training_setting & gd_setting = static_cast<const gd_training_setting &>(setting);

    gd_optimizer optimizer(gd_setting.learning_rate, gd_setting.decay_rate, gd_setting.max_epoches);

    VectorType p = net.get_parameter();
    NumericType fval = 0;
    tie(fval, p) = optimizer.optimize(net, p, X, y);

    net.set_parameter(p);

    return net;

}
