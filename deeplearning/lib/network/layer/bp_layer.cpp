#include <deeplearning/deep/bp_layer.h>

using namespace deeplearning;


int bp_layer::get_input_dim()
{
	return imput_dim;
}


int bp_layer::get_output_dim()
{
	return output_dim;
}

MatrixMapType bp_layer::get_W()
{
	return W;
}


MatrixMapType bp_layer::get_diff_W()
{
	return diff_W;
}

VectorMapType bp_layer::get_b()
{
	return b;
}

VectorMapType bp_layer::get_diff_b()
{
	return diff_b;
}

MatrixType bp_layer::get_activation()
{
	if (!record_activation)
	{
		throw runtime_error("the activation is not recorded");
	}
	return activation;
}


MatrixType bp_layer::get_output()
{
	if (!record_output)
	{
		throw runtime_error("the output is not recorded");
	}
	return output;
}

bp_layer::bp_layer(int input_dim_, int output_dim_, bool record_output_, bool record_activation_)
: input_dim(input_dim_), output_dim(output_dim_),record_output(record_output_),record_activation(record_activation_)
{

}


void bp_layer::compute_gredient(const MatrixType & delta)
{

	//dWb_mat = delta*[self.layered_output{2*num_maps}', ones(N,1)];

#if defined USE_PARTIAL_GPU
	GPUMatrixType gDelta = delta, gInput = input, gDiffw;
	GPUVectorType gDiffb;

	gDiffw = gDelta*gInput.transpose();
	gDiffb = gDelta.rowwise().sum();

	diff_W = (MatrixType) gDiffw;
	diff_b = (VectorType) gDiffb;

#else
	diff_W.noalias() = delta*input.transpose();
	diff_b.noalias() = delta.rowwise().sum();
#endif


}


NumericType bp_layer::cost(const MatrixType & x)
{
	return 0;
}
MatrixType bp_layer::gradient(const MatrixType & x)
{
	return MatrixType::Zero(x.rows(), x.cols());
}
