#include <artifact/network/layer/logisitic_layer.h>
#include <artifact/util/matrix_util.h>

using namespace artifact::network;


logistic_layer::logistic_layer(int input_dim, int output_dim)
:mlp_layer( input_dim, output_dim)
{
}

MatrixType logistic_layer::predict(const MatrixType & X)
{

	input = X;
#if defined USE_PARTIAL_GPU
	GPUMatrixType gW = W, gX = X, gDest;
	GPUVectorType gb = b;
	gDest = gW*gX;


	gDest.colwise() += gb;

	gDest = logistic_func(gDest);

	return (MatrixType)gDest;
#else

	MatrixType dest;
	dest.noalias() = W*X;

	dest.colwise() += b;
	dest = logistic_func(dest);//(1 + (-dest).array().exp()).inverse();
	return dest;

#endif
}

VectorType logistic_layer::predict(const VectorType & x)
{


#if defined USE_PARTIAL_GPU
	GPUMatrixType gW = W, gX = X, gDest;
	GPUVectorType gb = b;
	gDest = gW*gX;


	gDest.colwise() += gb;

	gDest = logistic_func(gDest);

	return (MatrixType)gDest;
#else

	VectorType dest;
	dest.noalias() = W*x;

	dest.colwise() += b;
	dest = logistic_func(dest);//(1 + (-dest).array().exp()).inverse();
	return dest;

#endif
}


MatrixType logistic_layer::compute_delta(const MatrixType & input, const MatrixType & output)
{
    //output_delta = 2/N*((Reconstruction - X).*Reconstruction.*(1-Reconstruction));

    if (not object)
    {
        throw runtime_error("The layer is not assigned with an objective.");
    }
    MatrixType result = 1-output.array();
    result.array()*= output.array();
    result.array() *= object->cost_gradient(output).array();
    return result;
}

MatrixType logistic_layer::backprop_delta(const MatrixType & delta, const MatrixType & input, const MatrixType & output)
{

    MatrixType new_delta =  (W.transpose()*delta);
    new_delta.array() *= input.array();
    new_delta.array() *= (1 - input.array());

    return new_delta;

}
