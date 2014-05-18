
#include "config.h"
#include <math.h>

#include "cumath.h"

__global__ void _gausian_kernel(  psFloat* val, psFloat*d, psFloat* d2, psFloat* x , int M, psFloat* param, CalculationType calType)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= M)
		return;
	
	val[i] = exp(-x[i]*x[i]/(2*param[i]*param[i]));
	
	if(calType != Eval)
	{
		d[i] = - val[i]/(2*param[i]*param[i]);
		d2[i] = val[i]/(4*pow(param[i],4));
	
	}
}


void gausian_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* x, int M, psFloat* param, CalculationType calType)
{
	int numGrid = (M + 512 -1)/512;
	_gausian_kernel<<<numGrid,512>>>(val, d, d2, x,M, param, calType);
}


//对X的每i行做以param[i]为参数的Gaussian核

__global__ void _rowwise_gausian_kernel(  psFloat* val, psFloat*d, psFloat* d2, psFloat* X , int M, int N,  psFloat* param, CalculationType calType)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= M || j > N)
		return;
	
	val[i*N + j] = exp(-X[i*N + j]*X[i*N + j]/(2*param[i]*param[i]));
	
	if(calType != Eval)
	{
		d[i*N + j]  = - val[i*N + j] /(2*param[i]*param[i]);
		d2[i*N + j]  = val[i*N + j] /(4*pow(param[i],4));
	
	}
}


void rowwise_gausian_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* X, int M, int N, psFloat* param, CalculationType calType)
{
	dim3 dimBlock(16,32);
	dim3 dimGrid((M +dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
	_rowwise_gausian_kernel<<<dimGrid,dimBlock>>>(val, d, d2, X,M,N, param, calType);
}



__global__ void _quadratic_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* x , int M, psFloat* param, CalculationType calType)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= M)
		return;

	psFloat cur_val = 1 - x[i]*x[i]/(param[i]*param[i]);
	
	if (cur_val < 0) cur_val = 0;
	
	val[i]  = cur_val*cur_val;
	
	if(calType != Eval)
	{
		d[i] = - 2*cur_val/(param[i]*param[i]);
		d2[i] = 2/(pow(param[i],4));
	
	}
}



void quadratic_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* x, int M, psFloat* param, CalculationType calType)
{
	int numGrid = (M + 512 -1)/512;
	_quadratic_kernel<<<numGrid,512>>>(val, d, d2, x,M, param, calType);
}



//对X的每i行做以param[i]为参数的Quadratic核

__global__ void _rowwise_quadratic_kernel(  psFloat* val, psFloat*d, psFloat* d2, psFloat* X , int M, int N,  psFloat* param, CalculationType calType)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= M || j > N)
		return;
	
	psFloat cur_val = 1 - X[i*N + j]*X[i*N + j]/(param[i]*param[i]);
	
	if (cur_val < 0) cur_val = 0;
	
	val[i*N + j]  = cur_val*cur_val;
	
	if(calType != Eval)
	{
		d[i*N + j] = - 2*cur_val/(param[i]*param[i]);
		d2[i*N + j] = 2/(pow(param[i],4));
	
	}
}


void rowwise_quadratic_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* X, int M, int N, psFloat* param, CalculationType calType)
{
	dim3 dimBlock(16,32);
	dim3 dimGrid((M +dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
	_rowwise_quadratic_kernel<<<dimGrid,dimBlock>>>(val, d, d2, X,M,N, param, calType);
}

