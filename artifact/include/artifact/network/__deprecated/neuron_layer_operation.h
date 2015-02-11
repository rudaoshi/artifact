/*
 * neuron_layer_operation.cpp
 *
 *  Created on: 2010-6-11
 *      Author: sun
 */
#include <liblearning/core/config.h>

#include <liblearning/util/math_util.h>
#include <liblearning/util/matrix_util.h>
#ifdef USE_MKL
	#include <mkl_blas.h>
	#include <mkl_vml_functions.h>

	#include <mkl_vsl.h>
//#elif defined USE_GPU
//
//	#include <gpumatrix/CORE>

#endif



void save_matrix(const string & filename, const MatrixType & perf);

namespace deep
{



	template <typename Dest, typename MT, typename MT2, typename VT>
	void linear_layer_output(Dest & dest, const MT & W, const char Trans,  const MT2 & X, const VT & b)
	{

	#if defined USE_MKL
		//do linear level
		MatrixType Y(W.rows(),X.cols());
		matrix_linear_transform(Y.data(), W.rows(),W.cols(), X.cols(), 1.0, W.data(), Trans, X.data(), 1, b.data());
	#elif defined USE_PARTIAL_GPU
		GPUMatrixType gW = W, gX = X, gDest;
		GPUVectorType gb = b;

		if (Trans == 'T')
		{
			gDest =  gW.transpose()*gX;
		}
		else
		{
			gDest = gW*gX;
		}

		gDest.colwise() += gb;

		dest = (Dest)gDest;

	#else
		if (Trans == 'T')
		{

			dest.noalias() =  W.transpose()*X;
		}
		else
		{
			dest.noalias() = W*X;
		}

		dest.colwise() += b;
		//for (int i = 0; i < dest.cols(); i++)
		//{
		//	dest.col(i) += b;
		//}
	#endif



	}

	#ifdef USE_MKL
	template <typename MT, typename VT>
	void linear_layer_output(MatrixType & Y, const MT * W,const char Trans, const VT & b, const MatrixType & X)
	{
		matrix_linear_transform(Y.data(), W.rows(),W.cols(), X.cols(), 1.0, W.data(), Trans, X.data(), 1, b.data());

	}
	#endif

	template <typename Dest, typename MT, typename MT2, typename VT>
	void logistic_layer_output(Dest & dest, const MT & W, const char Trans, const MT2 & X, const VT & b)
	{

	#ifdef USE_MKL
		//do linear level
		MatrixType Y(W.rows(),X.cols());
		matrix_logistic_transform(Y.data(), W.rows(),W.cols(), X.cols(), -1.0, W.data(), Trans, X.data(), -1, b.data());

	#elif defined USE_PARTIAL_GPU
		GPUMatrixType gW = W, gX = X, gDest;
		GPUVectorType gb = b;

		if (Trans == 'T')
		{
			gDest =  gW.transpose()*gX;
		}
		else
		{
			gDest = gW*gX;
		}

		gDest.colwise() += gb;

		gDest = logistic_func(gDest);

		dest = (Dest)gDest;
	#else

		linear_layer_output(dest,W, Trans,  X, b);
		dest = logistic_func(dest);//(1 + (-dest).array().exp()).inverse();

	#endif

	}

	template <typename Dest, typename MT, typename MT2, typename VT>
	void logistic_layer_output(Dest & dest, Dest & activation, const MT & W, const char Trans, const MT2 & X, const VT & b)
	{

	#ifdef USE_MKL
		//do linear level
		MatrixType Y(W.rows(),X.cols());
		matrix_logistic_transform(Y.data(), W.rows(),W.cols(), X.cols(), -1.0, W.data(), Trans, X.data(), -1, b.data());

	#elif defined USE_PARTIAL_GPU
		GPUMatrixType gW = W, gX = X, gDest;
		GPUVectorType gb = b;

		if (Trans == 'T')
		{
			gDest =  gW.transpose()*gX;
		}
		else
		{
			gDest = gW*gX;
		}


		gDest.colwise() += gb;


		activation = (Dest)gDest;

		gDest = logistic_func(gDest);

		dest = (Dest)gDest;
	#else

		linear_layer_output(dest,W, Trans,  X, b);
		activation = dest;
		dest = logistic_func(dest);//(1 + (-dest).array().exp()).inverse();

	#endif

	}

	#ifdef USE_MKL
	template <typename MT, typename VT>
	void logistic_layer_output(MatrixType & Y, const MT * W,const char Trans, const VT & b, const MatrixType & X)
	{

		matrix_logistic_transform(Y.data(), W.rows(),W.cols(), X.cols(), -1.0, W.data(), Trans, X.data(), -1, b.data());

	}
	#endif

	template <typename MT, typename MT2, typename VT>
	void backprop_diff(MT & diffw, VT & diffb, const MT2 & input, const MT2 & delta)
	{
		//dWb_mat = delta*[self.layered_output{2*num_maps}', ones(N,1)];
	#ifdef USE_MKL
		NumericType * dw = const_cast<NumericType *>(diffw.data());
		matrix_dot(dw,delta.rows(),delta.cols(),input.rows(),input.cols(),1.0, delta.data(), 'N',input.data(),'T');
		NumericType * db = const_cast<NumericType *>(diffb.data());
		extern NumericType ONES[];
		vector_linear_transform(db, delta.rows(),delta.cols(), 1.0, delta.data(), 'N', 	ONES);

	#elif defined USE_PARTIAL_GPU
		GPUMatrixType gDelta = delta, gInput = input, gDiffw;
		GPUVectorType gDiffb;

		gDiffw = gDelta*gInput.transpose();
		gDiffb = gDelta.rowwise().sum();

		diffw = (InterfaceMatrixType) gDiffw;
		diffb = (InterfaceVectorType) gDiffb;

	#else
		diffw.noalias() = delta*input.transpose();
		diffb.noalias() = delta.rowwise().sum();
	#endif


	}

	template <typename MT, typename MT2>
	void linear_delta_update(MT2 & delta, const MT & W, const MT2 & input)
	{
		//delta = W{level}'*delta;
	#ifdef USE_MKL
		MatrixType new_delta(W.cols(),delta.cols());

		matrix_dot(new_delta.data(),W.rows(),W.cols(),delta.rows(),delta.cols(),1.0, W.data(), 'T', delta.data(),'N');
		return new_delta;
	#elif defined USE_PARTIAL_GPU
		GPUMatrixType gDelta = delta, gInput = input, gW = W,gResult;
		gResult = gW.transpose()*gDelta;
		delta = (MT2) gResult;
	#else
		delta =  W.transpose()*delta;

	#endif



	}



	//
	template <typename MT>
	MT logistic_delta( const MT & output, const MT & error_diff)
	{
		//output_delta = 2/N*((Reconstruction - X).*Reconstruction.*(1-Reconstruction));
	#ifdef USE_MKL
		MatrixType delta(X.rows(),X.cols());

		int N = X.cols();
		int size = X.size();

		NumericType * TEMP = new NumericType[size];

		extern NumericType ONES[];

		vdSub(size,Recon.data(),X.data(),TEMP);
		vdMul(size,TEMP,Recon.data(),TEMP);

		vdSub(size,ONES,Recon.data(),delta.data());
		vdMul(size,TEMP,delta.data(),delta.data());

		NumericType coe = 2.0/N;
		int one_i = 1;

		dscal(&size,&coe, delta.data(),& one_i);
		// delta = 2/N*((Reconstruction - X).*Reconstruction.*(1-Reconstruction));

		delete TEMP;
		return delta;
	#elif defined USE_PARTIAL_GPU
		GPUMatrixType gOutput = output, gErrorDiff = error_diff, gResult;
		gResult = (gErrorDiff.array()*gOutput.array())*(1-gOutput.array());

		return (MT) gResult;


	#else
		MatrixType result = 1-output.array();
		result.array()*= output.array();

		//std::cout << "[ " << result.rows() << "x" << result.cols() << "]" <<std::endl;
		//std::cout << "[ " << error_diff.rows() << "x" << error_diff.cols() << "]" <<std::endl;
		result.array() *= error_diff.array();
		return result;
	#endif
	}


	template <typename MT, typename MT2>
	void logistic_delta_update(MT2 & delta, const MT & W, const MT2 & input)
	{
		//		delta = (W{map+1}'*delta).* self.layered_output{map+1}.*(1-self.layered_output{map+1});

	#ifdef USE_MKL
		matrix_dot(new_delta.data(),W.rows(),W.cols(),delta.rows(),delta.cols(),1.0, W.data(), 'T', delta.data(),'N');

		extern NumericType TEMP[];
		extern NumericType ONES[];
		int N  = new_delta.size();
		vdSub(N,ONES,input.data(),TEMP);
		vdMul(N,TEMP,input.data(),TEMP);
		vdMul(N,TEMP,new_delta.data(),new_delta.data());
		return new_delta;

	#elif defined USE_PARTIAL_GPU
		GPUMatrixType gDelta = delta, gInput = input, gW = W, gResult;

		//std::cout <<gDelta.squaredNorm();
		//std::cout << gInput.squaredNorm() << std::endl;
		//std::cout << gW.squaredNorm() << std::endl;

		gResult = (gW.transpose()*gDelta);
		gResult.array() *= gInput.array();
		gResult.array() *= (1 - gInput.array());
		delta =  (MT2) gResult ;

	#else
		delta =  (W.transpose()*delta);
		delta.array() *= input.array();
		delta.array() *= (1 - input.array());
	#endif
	}

	template< typename MT>
	void rbm_W_update(MT & W_inc, const MT & X0, const MT & Y0, const MT & X1, const MT & Y1, const MT & W,
		NumericType cur_momentum, NumericType epsilonw, NumericType weightcost)
	{
		int N = X0.cols();

	#if defined USE_PARTIAL_GPU
		GPUMatrixType gX0 = X0, gY0 = Y0, gX1 = X1, gY1 = Y1, gW = W, gW_inc = W_inc, gNewW_inc;

		gNewW_inc = cur_momentum*gW_inc + 	epsilonw*( (gY0*gX0.transpose()-gY1*gX1.transpose())/N - weightcost*gW);

		W_inc = (MT) gNewW_inc;

	#else
		W_inc *= cur_momentum;

		W_inc += epsilonw*( (Y0*X0.transpose()-Y1*X1.transpose())/N - weightcost*W);
//		W_inc = cur_momentum*W_inc + 	epsilonw*( (Y0*X0.transpose()-Y1*X1.transpose())/N - weightcost*W);

	#endif
	}

}
