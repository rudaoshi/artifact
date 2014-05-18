#include <deeplearning/network/logistic_layer.h>
#include <deeplearning/util/matrix_util.h>



logistic_layer::logistic_layer input_dim, int output_dim, bool record_output_ , bool record_activation_ )
:bp_layer( input_dim, output_dim, record_output_,record_activation_)
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

MatrixType logistic_layer::compute_delta()
{
	//output_delta = 2/N*((Reconstruction - X).*Reconstruction.*(1-Reconstruction));

#if defined USE_PARTIAL_GPU
	GPUMatrixType gOutput = output, gErrorDiff = error_diff, gResult;

	gResult = 1-gOutput.array();
	gResult.array() *= gOutput.array();
	gResult.array() *= gErrorDiff.array();

	return (MatrixType) gResult;


#else
	MatrixType result = 1-output.array();
	result.array()*= output.array();
	result.array() *= self.gredient(input).array();
	return result;
#endif
}

void logistic_layer::backprop_delta(MatrixType & delta)
{

#if defined USE_PARTIAL_GPU
	GPUMatrixType gDelta = delta, gInput = input, gW = W, gResult;

	gResult = (gW.transpose()*gDelta);
	gResult.array() *= gInput.array();
	gResult.array() *= (1 - gInput.array());
	delta =  (MatrixType) gResult ;

#else
	delta =  (W.transpose()*delta);
	delta.array() *= input.array();
	delta.array() *= (1 - input.array());
#endif
}
