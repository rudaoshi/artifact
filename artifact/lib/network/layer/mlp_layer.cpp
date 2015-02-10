#include <algorithm>

#include <artifact/network/layer/mlp_layer.h>


using namespace artifact::network;


int mlp_layer::get_input_dim()
{
	return input_dim;
}


int mlp_layer::get_output_dim()
{
	return output_dim;
}



VectorType make_parameter(const MatrixType & W, const VectorType & b)
{
    auto parameter = VectorType::Zero(W.size() + b.size());

    std::copy(parameter.data(), W.data(), W.data() + W.size());
    std::copy(parameter.data() + W.size(), b.data(), b.size());

    return parameter;

}

void mlp_layer::set_parameter(const VectorType & parameter)
{
    assert( parameter.size() == W.size() + b.size());

    std::copy(W.data(), parameter.data(),  W.data() + W.size());
    std::copy(b.data(), parameter.data() + W.size(),  b.size());
}


mlp_layer::mlp_layer(int input_dim_, int output_dim_)
    : input_dim(input_dim_), output_dim(output_dim_)
{

}

VectorType mlp_layer::compute_param_gradient(const MatrixType & delta, const MatrixType & input, const MatrixType & output)
{
    //dWb_mat = delta*[self.layered_output{2*num_maps}', ones(N,1)];

#if defined USE_PARTIAL_GPU
	GPUMatrixType gDelta = delta, gInput = input, gDiffw;
	GPUVectorType gDiffb;

	gDiffw = gDelta*gInput.transpose();
	gDiffb = gDelta.rowwise().sum();

	MatrixType diff_W = (MatrixType) gDiffw;
	VectorType diff_b = (VectorType) gDiffb;

#else
    MatrixType diff_W = delta*input.transpose();
    VectorType diff_b = delta.rowwise().sum();
#endif

    return make_parameter(diff_W, diff_b);

}


const VectorType  &mlp_layer::get_parameter()
{
    return make_parameter(this->W, this->b);
}

void mlp_layer::set_object(shared_ptr<objective> & object_ )
{
    object = object_;
}
