
#include <artifact/network/network_trainer.h>
#include <artifact/optimization/gd_optimizer.h>
#include <artifact/optimization/sgd_optimizer.h>

using namespace artifact::network;


optimization_trainer::optimization_trainer(optimizer & optimizer__)
:optimizer_(optimizer__)
{

}
deep_network optimization_trainer::train(deep_network & net,
        const MatrixType & X, const MatrixType * y)
{

    VectorType p = net.get_parameter();
    NumericType fval = 0;
    p = optimizer_.optimize(net, p, X, y);

    net.set_parameter(p);

    return net;

}
