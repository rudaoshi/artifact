/*
* deep_auto_encoder.cpp
*
*  Created on: 2010-6-3
*      Author: sun
*/

#include <artifact/network/deep_network.h>


using namespace artifact::network;

deep_network::deep_network()
{
}

deep_network::~deep_network()
{
}

MatrixType deep_network::predict(const MatrixType & X)
{

    MatrixType result = X;

    for (auto & layer: this->layers)
    {
        result = layer.predict(result);
    }

    return result;


}

RowVectorType deep_network::predict(const RowVectorType & x)
{

    RowVectorType result = x;

    for (auto & layer: this->layers)
    {
        result = layer.predict(result);
    }

    return result;


}
    
vector<pair<MatrixType, MatrixType>> deep_network::feed_forward(const MatrixType &input)
{
    vector<pair<MatrixType, MatrixType>> result;
    MatrixType cur_layer_input = input;
    for (auto & layer: this->layers)
    {
        auto feed_forward_info = layer.predict_with_activator(cur_layer_input);
        result.push_back(feed_forward_info);
        cur_layer_input = feed_forward_info.second;

    }

    return result;
}
#include <iostream>
vector<pair<MatrixType, RowVectorType>> deep_network::back_propagate(const MatrixType & input,
        const MatrixType & y,
        const vector<pair<MatrixType, MatrixType>> & laywise_output)
{

    MatrixType loss_gradient;

    vector<pair<MatrixType, RowVectorType>> gradients(layers.size());

    for (int i = layers.size() - 1; i >= 0; i --)
    {
        const MatrixType * cur_input = 0;

        if (i > 0)
            cur_input = &laywise_output[i-1].second;
        else
            cur_input = &input;

        const MatrixType * cur_ouput = & laywise_output[i].second;
        const MatrixType * cur_activator = & laywise_output[i].first;

        mlp_layer * layer = &this->layers[i];
        if (layer->is_loss_contributor())
        {
            MatrixType cur_loss_gradient = layer->compute_loss_gradient(*cur_ouput, y);

            if (loss_gradient.size() == 0)
            {
                loss_gradient = cur_loss_gradient;
            }
            else
            {
                loss_gradient += cur_loss_gradient;
            }

        }
        MatrixType delta = layer->compute_delta(*cur_activator, *cur_ouput, loss_gradient);

        gradients[i] = layer->compute_param_gradient(
                *cur_input, delta);


        loss_gradient = layer->backprop_loss_gradient(delta);


    }

    return gradients;
}


NumericType deep_network::objective(const MatrixType & X,
        const MatrixType * y)
{
    /**
    * Currently, objective at last layer is supported;
    */
    MatrixType output = this->predict(X);
    auto last_layer = this->layers.rbegin();

    return last_layer->loss_func->loss(output, y);

}


tuple<NumericType, VectorType> deep_network::gradient(const MatrixType & x,
        const MatrixType * y)
{

    vector<pair<MatrixType, MatrixType>> forward_result =  deep_network::feed_forward(x);


    vector<pair<MatrixType, RowVectorType>> backprop_result = deep_network::back_propagate(x, *y,
            forward_result);


    auto & last_layer_output = forward_result.rbegin()->second;
    auto last_layer = this->layers.rbegin();

    NumericType loss =  last_layer->loss_func->loss(last_layer_output, y);

    int total_param_num = 0;
    for (auto & layer : this->layers)
    {
        total_param_num += (layer.get_input_dim() + 1) * layer.get_output_dim();
    }

    VectorType Wb_gradient = VectorType::Zero(total_param_num);

    int start_idx = 0;
    for (auto & Wb_pair : backprop_result)
    {
        Map<MatrixType>(Wb_gradient.data()+ start_idx, Wb_pair.first.rows(),Wb_pair.first.cols()) = Wb_pair.first;
        start_idx += Wb_pair.first.size();
        Map<VectorType>(Wb_gradient.data()+ start_idx, Wb_pair.second.size()) = Wb_pair.second;
        start_idx += Wb_pair.second.size();
    }
    return make_tuple(loss, Wb_gradient);
}



VectorType deep_network::get_parameter() {

    int total_param_num = 0;
    for (auto & layer : this->layers)
    {
        total_param_num += (layer.get_input_dim() + 1) * layer.get_output_dim();
    }

    VectorType Wb = VectorType::Zero(total_param_num);

    int start_idx = 0;
    for (auto & layer : this->layers)
    {
        Map<MatrixType>(Wb.data()+ start_idx,layer.get_input_dim(),layer.get_output_dim()) = layer.W;
        start_idx += layer.get_input_dim()  * layer.get_output_dim();
        Map<VectorType>(Wb.data()+ start_idx, layer.get_output_dim()) = layer.b;
        start_idx += layer.get_output_dim();
    }
    return Wb;
}

void deep_network::set_parameter(const VectorType &parameter_)
{
    int start_idx = 0;
    for (auto & layer : this->layers)
    {
        layer.W = Map<const MatrixType>(parameter_.data()+ start_idx,layer.get_input_dim(),layer.get_output_dim()) ;
        start_idx += layer.get_input_dim()  * layer.get_output_dim();
        layer.b = Map<const VectorType>(parameter_.data()+ start_idx, layer.get_output_dim());
        start_idx += layer.get_output_dim();
    }

}

void deep_network::add_layer(const mlp_layer &layer)
{
    layers.push_back(layer);
}

void deep_network::remove_layer(int pos)
{
    layers.erase(layers.begin() + pos);
}

mlp_layer & deep_network::get_layer(int pos)
{
    return layers[pos];
}


int deep_network::get_layer_num()
{
    return layers.size();
}
