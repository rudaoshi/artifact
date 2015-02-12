
#include <artifact/network/network_creator.h>
#include <artifact/network/layer/activator.h>
#include <artifact/network/deep_network.h>
#include <artifact/loss/loss.h>
using namespace artifact::network;
using namespace artifact::loss;


deep_network random_network_creator::create(const network_architecture & architec_param,
                    const create_context * context = 0)
{
    if (architec_param.layer_sizes.size() != architec_param.activator_types.size() + 1)
    {
        throw runtime_error("Bad network architecture setting!");
    }

    deep_network network;

    for (int i = 0; i < architec_param.activator_types.size(); i++)
    {
        int input_num = architec_param.layer_sizes[i];
        int output_num = architec_param.layer_sizes[i+1];

        string type = architec_param.activator_types[i];

        shared_ptr<activator> activate_func;
        NumericalType random_scale = 0;
        if (type == linear_activator::type)
        {
            activate_func = new linear_activator();
            random_scale = 4.0 * sqrt(6.0 / (input_num + output_num));
        }
        else if (type == logistic_activator::type)
        {
            activate_func = new logistic_activator();
            random_scale = 4.0 *sqrt(6.0 / (input_num + output_num));
        }
        else
        {
            throw runtime_error("Unkown activation function: " + type);
        }
        //        else if (type == tanh_activator::type)
//        {
//            activate_func = new tanh_activator();
//            random_low = -sqrt(6.0 / (input_num + output_num));
//            random_high = sqrt(6.0 / (input_num + output_num));
//        }
//        else if (type == softmax_activator::type)
//        {
//            activate_func = new softmax_activator();
//            random_low = 0.0;
//            random_high = -0.0;
//        }

        mlp_layer layer(input_num, output_num, activate_func);

        layer.W = MatrixType::Random(input_num, output_num) * random_scale;
        layer.b = MatrixType::Zero(output_num);

        network.add_layer(layer);


    }

    if (architec_param.loss == "mse")
    {
        shared_ptr<loss> loss = new mse_loss();
        network.get_layer(architec_param.activator_types.size()-1).loss = loss;
    }

    return network;

}

#endif