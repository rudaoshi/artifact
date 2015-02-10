
#include <math.h>
#include <cublas.h>
#include <cutil.h>

#include "config.h"
#include "cumath.h"
#include "util.h"

#include "spps.h"

namespace KernelRegressionManifold
{

	psFloat * _Distance,* _B, * _P, * _Q;  //nS * nX matrix
	psFloat * _temp_nS_nX_1, * _temp_nS_nX_2, * _temp_nS_nX_3;             // nS * nX matrix
	psFloat * _MX ; // Dim * nX   matrix
	psFloat * _temp_nX_1;  // nX  array
	psFloat * _temp_dim_nX_1;  // dim * nX matrix



	void slps_regkernel(psFloat * kernelVal, psFloat * kernelD, psFloat* kernelD2, psFloat * S, int nS,  psFloat * X, int nX, int dim,  psFloat * sigma, KernelType kernel, CalculationType calType)
	{

		// static * distance;
		pairwise_distance(S, nS, nS*sizeof(psFloat), X, nX*sizeof(psFloat), nX, dim, _Distance);

	#ifdef _DEBUG_DETAIL
		display("the resulting normDiffV matrix is:", _Distance, nS ,  nX);
	#endif

		if (kernel == Gaussian)
		{
			rowwise_gausian_kernel( kernelVal, kernelD, kernelD2, _Distance, nS, nX, sigma, calType);

		}
		else if(kernel == Quadratic)
		{
			rowwise_quadratic_kernel( kernelVal, kernelD, kernelD2,_Distance, nS, nX, sigma, calType);

		}

	#ifdef _DEBUG_DETAIL
		display("the resulting kernelVal matrix is:", kernelVal,nS ,  nX);
	#endif
		
	}




	void slps_kernelregbasis(psFloat * B, psFloat * P, psFloat * Q, psFloat * S, int nS,  psFloat * X, int nX, int dim,  psFloat * sigma, KernelType kernel, CalculationType calType)
	{
		
		// calculate the kernel value and kernel differences
		// kernelVal, kernelD, kernelD2  可以与 b, p, q 重用
		slps_regkernel(B,P,Q, S,  nS,  X, nX, dim, sigma,kernel, calType);

		psFloat * _sumval = _temp_nX_1;

		column_wise_sum(_sumval, B, nS, nX);
		column_wise_normal(B,B,nS,nX, _sumval);
		
	#ifdef _DEBUG_DETAIL
		display("the resulting b matrix is:", B,nS ,  nX);
	#endif
		
		if (calType == Eval)
			return;
				
		// Calculate p
		column_wise_normal(P,P,nS,nX, _sumval);
		cublasFscal (nS*nX, -2, P, 1);
	#ifdef _DEBUG_DETAIL
		display("the resulting p matrix is:", P, nS ,  nX); 
	#endif
		// Calculate q
		column_wise_normal(Q,Q,nS,nX, _sumval);
		cublasFscal (nS*nX, 4, Q, 1);
	#ifdef _DEBUG_DETAIL
		display("the resulting q matrix is:", Q, nS ,  nX); 
	#endif


	}

	// Calculate object value (mX) of X throught the mapping constructed via Kernel regression from S to T
	void slps_map(psFloat * mX, psFloat * X, int nX, psFloat * T, int Dim, psFloat * S, int nS, int dim,    psFloat * sigma, KernelType kernel, CalculationType calType)
	{
		slps_kernelregbasis(_B, _P, _Q,  S, nS,  X, nX,  dim, sigma, kernel, calType);
		// Compute mx
		// ps.mx = ps.T*ps.b;
		cublasFgemm('N','N',Dim,  nX, nS, 1, T, Dim, _B, nS, 0, mX, Dim);			
	}

	void slps_dist(psFloat * dist, psFloat * JDX,  psFloat * X, int nX, psFloat * Y,  psFloat * T, int Dim, psFloat * S, int nS,  int dim,  psFloat * sigma, KernelType kernel, CalculationType calType)
	{

		slps_map(_MX, X, nX, T, Dim, S, nS,  dim, sigma, kernel, calType);
		
		// tempD = y - ps.mx;
		// dist = norm(tempD);
		// MX =  MX - Y
		cublasFaxpy(Dim*nX, -1, Y, 1, _MX, 1);

		if (calType == Eval || dist != 0)
		{
			column_wise_squared_norm2(dist, _MX, Dim, nX);
			
		}
		if (calType == Eval)
			return;
		
		//Jdx_i = 2*Jmx_i*(mx_i-y_i) = 2*Jbx*T'*(mx_i-y_i) = 2*diffVecs_i*(diag(p_i)-p_i*b_i')T'*(mx_i-y_i)
		// Jdx_i = 2*[(S-repmat(x_i,1,nS))*(diag(p_i)*T'*(mx_i-y_i)-p_i*b_i'*T'*(mx_i-y_i))]
		// _temp_nS_nX_i = T'*(mx_i-y_i);
		cublasFgemm('T','N',nS,nX,Dim, 1, T, Dim, _MX , Dim, 0, _temp_nS_nX_1, nS);
		// _temp_nX_i = b_i'*T'*(mx_i-y_i);
		column_wise_dot(_temp_nX_1, _B, _temp_nS_nX_1 , nS , nX);
		// temp_nS_nX2_i = p_i*b_i'*T'*(mx_i-y_i) 
		column_wise_scal(_temp_nS_nX_2, 1, _P, nS, nX, _temp_nX_1);
		
		// _temp_nS_nX3_i = diag(p_i)*T'*(mx_i-y_i)
		element_wise_scal(_temp_nS_nX_3, 1, _P , _temp_nS_nX_1,nS, nX);
		
		// _temp_nS_nX3_i = (diag(p_i)*T'*(mx_i-y_i)-p_i*b_i'*T'*(mx_i-y_i));
		cublasFaxpy(nS*nX, -1, _temp_nS_nX_2, 1, _temp_nS_nX_3, 1);
		
		// JDX = 2*S*_temp_nS_nX_3;
		cublasFgemm('N','N',dim,nX,nS, 2, S, dim, _temp_nS_nX_3 ,nS, 0, JDX, dim);
		
		// _temp_dim_nX_i = rempmat(x_i,1,ns)*_temp_nS_nX3_i = sum(_temp_nS_nX3_i)*x_i;
		column_wise_sum(_temp_nX_1, _temp_nS_nX_3 ,nS , nX);
		column_wise_scal(_temp_dim_nX_1,2, X, dim, nX, _temp_nX_1);
		
		cublasFaxpy(dim*nX, -1, _temp_dim_nX_1, 1, JDX, 1);
		
	}


	namespace _OptimHelper
	{

		psFloat * Y;
		int nY;
		psFloat * T;
		int Dim; 
		psFloat * S;
		int nS;
		int dim;
		psFloat * Sigma;
		KernelType kernel;
		// 优化辅助函数,用于求某点y到主曲面的投影指标
		// 本函数用于求解某点y到S(x)的距离
		// 使用前必须将y拷贝到变量y中。
		void slps_dist_help(psFloat * Dist, psFloat* X )
		{

		//	cublasFcopy(s.d, x, 1, x,1);

			slps_dist(Dist, 0,  X, nY,  Y,  T, Dim,  S, nS,  dim,  Sigma, kernel, Eval);

		//	static int i = 0;
			
		//	i ++;
			
		//	if (i % 10 == 0)
		#ifdef _DEBUG_DETAIL
//			display("computing the object value at point:", _x, 1 , s.d);
		#endif

		}

		// 优化辅助函数,用于求某点y到主曲面的投影指标
		// 本函数用于求解某点y到S(x)的距离对x的导数
		// 使用前必须将y拷贝到变量y中，并且slps_dist_help在之前紧接着被调用过。
		void slps_dist_Jacobbi_help(psFloat*  _Jdx, psFloat* _x )
		{

			slps_dist(0, JDX,  X, nY,X, nY,  Y,  T, Dim,  S, nS,  dim,  Sigma, kernel, Jacobbi);

		#ifdef _DEBUG_DETAIL	
			//display("computing the Jacobbi matrix at point:", _x, 1 , s.d);
			//display("the resulting Jacobbi matrix is:", _Jdx, 1 , s.d);
		#endif
			
		//	cublasFcopy(s.d, Jdx, 1, Jdx,1);
		}
	}

	void slps_project(psFloat * X, psFloat* Y, int nY,  bool withInitVal, psFloat* X0,  psFloat * T, int Dim, psFloat * S, int nS, int dim, psFloat * Sigma, KernelType kernel)
	{
		/**
		* 寻找到对应的初始点x
		*/
		_OptimHelper::Y = Y;
		_OptimHelper::nY = nY;
		_OptimHelper::T = T;
		_OptimHelper::Dim = Dim;
		_OptimHelper::S = S;
		_OptimHelper::nS = nS;
		_OptimHelper::dim = dim;
		_OptimHelper::Sigma = Sigma;
		_OptimHelper::kernel = kernel;
		
		if (!withInitVal)
		{
			pairwise_squared_distance( _Distance, S, nS, 0, X , nX , 0, dim);

			// 寻找tempM中最小值的位置。
			columnwise_min_index(_temp_nX_1, _Distance , nS, nX);
			copy_indexed_columns(X, _temp_nX_1, nX, S , dim, nS);
		}
		else
		{
			cublasFcopy(dim*nX, X0, 1, X,1);
		}

	#ifdef _DEBUG_DETAIL	
		display("The initial point is:", _x, 1 , s.d);
	#endif
		int iter;
		psFloat * fret = _temp_nX_1;
		psFloat ftol = 1e-6;
		
		// the resulting x is the projection index.
		batch_frprmn(nX, dim, X, ftol, &iter, fret, &slps_dist_help,&slps_dist_Jacobbi_help);
//		frprmn(s.d, _x, ftol, iter, fret, &slps_dist_help,&slps_dist_Jacobbi_help);

	}

	void kernelreg_init(int dim, int nS, int nX, int Dim )
	{

	//	psFloat * _Distance,* _B, * _P, * _Q;  //nS * nX matrix
	//psFloat * _temp_nS_nX_1, * _temp_nS_nX_2, * _temp_nS_nX_3;             // nS * nX matrix
	//psFloat * _MX ; // Dim * nX   matrix
	//psFloat * _temp_nX_1;  // nX  array
	//psFloat * _temp_dim_nX_1;  // dim * nX matrix

		cublasAlloc(nS*nX, sizeof(psFloat), (void**)& _Distance);
		cublasAlloc(nS*nX, sizeof(psFloat), (void**)& _B);
		cublasAlloc(nS*nX, sizeof(psFloat), (void**)& _P);
		cublasAlloc(nS*nX, sizeof(psFloat), (void**)& _Q);
		cublasAlloc(nS*nX, sizeof(psFloat), (void**)& _temp_nS_nX_1);
		cublasAlloc(nS*nX, sizeof(psFloat), (void**)& _temp_nS_nX_2);
		cublasAlloc(nS*nX, sizeof(psFloat), (void**)& _temp_nS_nX_3);

		cublasAlloc(Dim*nX, sizeof(int), (void**)& _MX);

		cublasAlloc(nX, sizeof(int), (void**)& _temp_nX_1);
		cublasAlloc(dim*nX, sizeof(int), (void**)& _temp_dim_nX_1);

	}

	void kernelreg_final(int dim, int nS, int nX, int Dim )
	{
		cublasFree( _Distance);
		cublasFree( _B);
		cublasFree( _P);
		cublasFree( _Q);
		cublasFree( _temp_nS_nX_1);
		cublasFree( _temp_nS_nX_2);
		cublasFree( _temp_nS_nX_3);

		cublasFree( _MX);

		cublasFree( _temp_nX_1);
		cublasFree( _temp_dim_nX_1);
	}
}
