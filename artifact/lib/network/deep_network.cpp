/*
* deep_auto_encoder.cpp
*
*  Created on: 2010-6-3
*      Author: sun
*/

#include <artifact/network/deep_network.h>
#include <cassert>

#include <artifact/util/math_util.h>
#include <artifact/util/matrix_util.h>
#include <artifact/network/restricted_boltzmann_machine.h>

#include <algorithm>
#include <artifact/network/neuron_layer_operation.h>

#include <artifact/network/network_optimobj_adapter.h>
#include <artifact/optimization/conjugate_gradient_optimizer.h>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/foreach.hpp>
#include <artifact/util/algo_util.h>


using namespace std;

using namespace artifact::network;



MatrixType deep_network::predict(const MatrixType & X)
{

    MatrixType result = X;

    for (auto layer: this->layers)
    {
        result = layer.predict(result);
    }

    return result;


}

VectorType deep_network::predict(const VectorType & x)
{

    VectorType result = x;

    for (auto layer: this->layers)
    {
        result = layer.predict(result);
    }

    return result;


}
    
vector<pair<MatrixType, MatrixType>> deep_network::feed_forward(const MatrixType &input)
{
    vector<pair<MatrixType, MatrixType>> result;
    MatrixType cur_layer_input = input;
    for (int i=0; i<layers.size(); i++)
    {
        result.push_back(layers[i].predict_with_activator(cur_layer_input));
    }

    return result;
}
    
vector<pair<MatrixType, VectorType>> deep_network::back_propagate(const MatrixType & input,
        const VectorType & y,
        const vector<pair<MatrixType, MatrixType>> & laywise_output)
{

    MatrixType delta;

    vector<pair<MatrixType, VectorType>> gradients(layers.size());

    for (int i = layers.size() - 1; i >= 0; i --)
    {
        MatrixType * cur_input = 0;

        if (i > 0)
            cur_input = &laywise_output[i-1].second;
        else
            cur_input = &input;

        MatrixType * cur_ouput = & laywise_output[i].second;
        MatrixType * cur_activator = & laywise_output[i].first;

        mlp_layer * layer = &this->layers[i];
        if (layer->is_loss_contributor())
        {
            if (delta.size() == 0) {
                delta = layer->compute_delta(*cur_activator, *cur_ouput, y);
            }
            else {
                delta += layer->compute_delta(*cur_activator, *cur_ouput, y);
            }
        }

        delta = layer->.backprop_delta(
                delta, *cur_activator);

        gradients[i] = layer->compute_param_gradient(
                *cur_input, delta);
    }

    return gradients;
}


NumericType deep_network::objective(const MatrixType & X,
        const VectorType & y)
{
    /**
    * Currently, objective at last layer is supported;
    */
    MatrixType output = this->predict(X);
    auto last_layer = this->layers.rbegin();

    return last_layer.loss_func.loss(output, y);

}

VectorType deep_network::gradient(const MatrixType & x,
        const VectorType & y)
{
    vector<pair<MatrixType, MatrixType>> deep_network::feed_forward(const MatrixType &input)
    {
        vector<pair<MatrixType, MatrixType>> result;
        MatrixType cur_layer_input = input;
        for (int i=0; i<layers.size(); i++)
        {
            result.push_back(layers[i].predict_with_activator(cur_layer_input));
        }

        return result;
    }

    vector<pair<MatrixType, VectorType>> deep_network::back_propagate(const MatrixType & input,
            const VectorType & y,
            const vector<pair<MatrixType, MatrixType>> & laywise_output)
}



VectorType deep_network::get_parameter() {

    int total_param_num = 0;
    for (auto layer : this->layers)
    {
        total_param_num += (layer.get_input_dim() + 1) * layer.get_output_dim();
    }

    VectorType Wb = VectorType::Zero(total_param_num);

    int start_idx = 0;
    for (auto layer : this->layers)
    {
        Map<MatrixType>(Wb.data()+ start_idx,layer.get_output_dim(),layer.get_input_dim()) = layer.W;
        start_idx += layer.get_input_dim()  * layer.get_output_dim();
        Map<VectorType>(Wb.data()+ start_idx, layer.get_output_dim()) = layer.b;
        start_idx += layer.get_output_dim();
    }
    return Wb;
}

void deep_network::set_parameter(const VectorType &parameter_)
{
    int start_idx = 0;
    for (auto layer : this->layers)
    {
        layer.W = Map<MatrixType>(parameter_.data()+ start_idx,layer.get_output_dim(),layer.get_input_dim()) ;
        start_idx += layer.get_input_dim()  * layer.get_output_dim();
        layer.b = Map<VectorType>(parameter_.data()+ start_idx, layer.get_output_dim());
        start_idx += layer.get_output_dim();
    }

}


deep_network::deep_network()
{
}

