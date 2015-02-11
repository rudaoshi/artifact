#include <artifact/network/layer/linear_layer.h>

using namespace artifact::network;


linear_layer::linear_layer(int input_dim,int output_dim)
    : mlp_layer( input_dim, output_dim)
{
}

MatrixType linear_layer::predict(const MatrixType & X)
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

	return dest;


#endif

}

VectorType linear_layer::predict(const VectorType & x)
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
	VectorType dest;
	dest.noalias() = W*x;

	dest.colwise() += b;

	return dest;


#endif

}

MatrixType linear_layer::compute_delta(const MatrixType & input,
        const MatrixType & output,
        const VectorType & y)
{
    if (not object)
    {
        throw runtime_error("The layer is not assigned with an objective.");
    }
    return loss_func->gradient(output, y);
}

MatrixType linear_layer::backprop_delta(const MatrixType & delta,
        const MatrixType & input,
        const MatrixType & output)
{
    //delta = W{level}'*delta;

#if defined USE_PARTIAL_GPU
	GPUMatrixType gDelta = delta, gInput = input, gW = W,gResult;
	gResult = gW.transpose()*gDelta;
	return (MatrixType) gResult;
#else
    return W.transpose()*delta;

#endif
}
