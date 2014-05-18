#include <liblearning/deep/linear_layer.h>

using namespace deep;


linear_layer::linear_layer（int input_dim, int output_dim, bool record_output_ , bool record_activation_ )
:backprop_neuron_layer( input_dim, output_dim, record_output_,record_activation_)
{
}

MatrixType linear_layer::output(const MatrixType & X)
{

#if defined USE_PARTIAL_GPU
	GPUMatrixType gW = W, gX = X, gDest;
	GPUVectorType gb = b;

	gDest = gW*gX;

	gDest.colwise() += gb;

	if (record_activation)
		activation = (MatrixType)gDest;

	if (record_output)
	{
		output = (MatrixType)gDest;
		return output;
	}
	else
	{
		return (MatrixType)gDest;
	}


#else
	MatrixType dest;
	dest.noalias() = W*X;

	dest.colwise() += b;

	if (record_activation)
		activation = dest;

	if (record_output)
	{
		output = dest;
	}

	return dest;


#endif

}

MatrixType linear_layer::delta(const MatrixType & output, const MatrixType & error_diff)
{
	return error_diff;
}

void linear_layer::delta_backprop(MatrixType & delta, const MatrixType & input)
{
	//delta = W{level}'*delta;

#if defined USE_PARTIAL_GPU
	GPUMatrixType gDelta = delta, gInput = input, gW = W,gResult;
	gResult = gW.transpose()*gDelta;
	delta = (MatrixType) gResult;
#else
	delta =  W.transpose()*delta;

#endif
}
