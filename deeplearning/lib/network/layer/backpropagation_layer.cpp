#include <algorithm>

#include <deeplearning/network/layer/backpropagation_layer.h>


using namespace deeplearning;


int backpropagation_layer::get_input_dim()
{
	return input_dim;
}


int backpropagation_layer::get_output_dim()
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

backpropagation_layer::backpropagation_layer(int input_dim_, int output_dim_)
    : input_dim(input_dim_), output_dim(output_dim_)
{

}

VectorType backpropagation_layer::compute_param_gradient(const MatrixType & delta, const MatrixType & input, const MatrixType & output)
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


const VectorType  &backpropagation_layer::get_parameter()
{
    return make_parameter(this->W, this->b);
}
void backpropagation_layer::set_parameter(const VectorType & parameter)
{
    assert( parameter.size() == W.size() + b.size());

    std::copy(W.data(), parameter.data(),  W.data() + W.size());
    std::copy(b.data(), parameter.data() + W.size(),  b.size());
}


NumericType backpropagation_layer::cost(const MatrixType & x)
{
	return 0;
}
MatrixType backpropagation_layer::cost_gradient(const MatrixType & x)
{
	return MatrixType::Zero(x.rows(), x.cols());
}
