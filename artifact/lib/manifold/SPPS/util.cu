
#include "config.h"
#include <stdio.h>


// 对矩阵A中任意一列y，求y+alpha*x之后，放入矩阵B的对应列中
__global__ void _column_wise_add(psFloat* B, psFloat alpha, psFloat* A, int r, int c,  psFloat beta, psFloat* x)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= r || j >= c)
		return;

	B[j*r+i] = alpha*A[j*r+i]+beta*x[i];
}

void column_wise_add(psFloat* B, psFloat alpha, psFloat* A, int r, int c,psFloat beta, psFloat* x)
{
	dim3 dimBlock(16,32);
	dim3 dimGrid((r+dimBlock.x-1)/dimBlock.x, (c+dimBlock.y-1)/dimBlock.y);

	_column_wise_add<<<dimGrid,dimBlock>>>(B,alpha, A, r, c, beta, x);
}

// 对矩阵A中第j列y，求y = x[j]*y之后，放入矩阵B的对应列中
__global__ void _column_wise_scal(psFloat* B, psFloat alpha, psFloat* A, int r, int c, psFloat* x)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= r || j >= c)
		return;

	B[j*r+i] = alpha* A[j*r+i]*x[j];
}

void column_wise_scal(psFloat* B, psFloat alpha, psFloat* A, int r, int c, psFloat* x)
{
	dim3 dimBlock(16,32);
	dim3 dimGrid((r+dimBlock.x-1)/dimBlock.x, (c+dimBlock.y-1)/dimBlock.y);
	_column_wise_scal<<<dimGrid,dimBlock>>>(B, alpha, A, r, c,  x);
}

// C[i,j] = A[i,j]*B[i,j]
__global__ void _element_wise_scal(psFloat* C, psFloat alpha, psFloat* A, psFloat * B, int r, int c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= r || j >= c)
		return;

	C[j*r+i] = alpha * A[j*r+i]*B[j*r+i];
}

void element_wise_scal(psFloat* C,psFloat alpha, psFloat* A, psFloat * B, int r, int c)
{
	dim3 dimBlock(16,32);
	dim3 dimGrid((r+dimBlock.x-1)/dimBlock.x, (c+dimBlock.y-1)/dimBlock.y);
	_element_wise_scal<<<dimGrid,dimBlock>>>(C, alpha, A, B, r, c);
}

// 对矩阵A中第j列y，求y = y/x[j]之后，放入矩阵B的对应列中
__global__ void _column_wise_normal(psFloat* B, psFloat* A, int r, int c, psFloat* x)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= r || j >= c)
		return;

	B[j*r+i] = A[j*r+i]/x[j];
}

void column_wise_normal(psFloat* B,psFloat* A, int r, int c, psFloat* x)
{
	dim3 dimBlock(16,32);
	dim3 dimGrid((r+dimBlock.x-1)/dimBlock.x, (c+dimBlock.y-1)/dimBlock.y);
	_column_wise_normal<<<dimGrid,dimBlock>>>(B, A, r, c,  x);
}

__global__ void _column_wise_sum(psFloat* n, psFloat* A, int r, int c)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	if (tid >= r || bid >=c)
		return;

	extern __shared__ psFloat s_data[];
	extern __shared__ psFloat curLen;

	s_data[tid] = A[bid*c+tid];

	__syncthreads();

	curLen = r;

	for (int i = (curLen+1)/2; i > 0 ; i = (curLen+1)/2 )
	{
		if(tid < i && tid+i < curLen)
		{
			s_data[tid]   += s_data[tid+i];
		}
		curLen = i;
		__syncthreads();
	}
	
	if (tid == 0)
		n[bid] = s_data[0];

}

void column_wise_sum(psFloat* n, psFloat* A, int r, int c)
{
	_column_wise_sum<<<c,r,r>>>(n, A, r, c);
}


__global__ void _column_wise_norm2(psFloat* n, psFloat* A, int r, int c)
{
	int bid = blockIdx.x ;
	int tid = threadIdx.x;



	// Add the overflow checking!
	
	if (tid >= r || bid >= c  )
		return;
		
	extern __shared__ psFloat s_data[]; 
	extern __shared__ psFloat curLen;

	
	s_data[tid] = A[bid*r+tid]*A[bid*r+tid];

	__syncthreads();

	curLen = r;

	for (int i = (curLen+1)/2; i > 0 ; i = (curLen+1)/2 )
	{
		if(tid < i && tid+i < curLen)
		{
			s_data[tid]   += s_data[tid+i];
		}
		curLen = i;
		__syncthreads();
	}
	
	if (tid == 0)
		n[bid] = sqrt(s_data[0]);

}


void column_wise_norm2(psFloat* n, psFloat* A, int r, int c)
{
	_column_wise_norm2<<<c,r,r>>>(n, A, r, c);
}


__global__ void _column_wise_dot(psFloat* n, psFloat* A, psFloat * B, int r, int c)
{
	int bid = blockIdx.x ;
	int tid = threadIdx.x;



	// Add the overflow checking!
	
	if (tid >= r || bid >= c  )
		return;
		
	extern __shared__ psFloat s_data[]; 
	
	s_data[tid] = A[bid*r+tid]*B[bid*r+tid];

	__syncthreads();

	curLen = r;

	for (int i = (curLen+1)/2; i > 0 ; i = (curLen+1)/2 )
	{
		if(tid < i && tid+i < curLen)
		{
			s_data[tid]   += s_data[tid+i];
		}
		curLen = i;
		__syncthreads();
	}
	
	if (tid == 0)
		n[bid] = s_data[0];

}


void column_wise_dot(psFloat* n, psFloat* A, psFloat * B, int r, int c)
{
	_column_wise_dot<<<c,r,r>>>(n, A, B, r, c);
}


__global__ void _column_wise_squared_norm2(psFloat* n, psFloat* A, int r, int c)
{
	int bid = blockIdx.x ;
	int tid = threadIdx.x;



	// Add the overflow checking!
	
	if (tid >= r || bid >= c  )
		return;
		
	extern __shared__ psFloat s_data[]; 
	extern __shared__ psFloat curLen;

	
	s_data[tid] = A[bid*r+tid]*A[bid*r+tid];

	__syncthreads();

	curLen = r;

	for (int i = (curLen+1)/2; i > 0 ; i = (curLen+1)/2 )
	{
		if(tid < i && tid+i < curLen)
		{
			s_data[tid]   += s_data[tid+i];
		}
		curLen = i;
		__syncthreads();
	}
	
	if (tid == 0)
		n[bid] = s_data[0];

}


void column_wise_squared_norm2(psFloat* n, psFloat* A, int r, int c)
{
	_column_wise_squared_norm2<<<c,r,r>>>(n, A, r, c);
}

__global__ void _column_wise_squared_distance_mm(psFloat* n, psFloat* A, psFloat * B, int r, int c)
{
	int bid = blockIdx.x ;
	int tid = threadIdx.x;

	// Add the overflow checking!
	
	if (tid >= r || bid >= c  )
		return;
		
	extern __shared__ psFloat s_data[]; 
	extern __shared__ psFloat curLen;

	
	s_data[tid] = (A[bid*r+tid]-B[bid*r+tid])*(A[bid*r+tid]-B[bid*r+tid]);

	curLen = r;

	for (int i = (curLen+1)/2; i > 0 ; i = (curLen+1)/2 )
	{
		if(tid < i && tid+i < curLen)
		{
			s_data[tid]   += s_data[tid+i];
		}
		curLen = i;
		__syncthreads();
	}
	
	if (tid == 0)
		n[bid] = sqrt(s_data[0]);
}

void column_wise_squared_distance_mm(psFloat* n, psFloat* A, psFloat* B, int r, int c)
{
	_column_wise_squared_distance_mm<<<c,r,r>>>(n, A, B, r, c);
}




__global__ void _column_wise_distance_mv(psFloat* dis, psFloat* A, int r, int c, psFloat* x)
{

	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (tid >= r ||   bid >= c)
		return;
		
	extern __shared__ psFloat s_data[];
	extern __shared__ psFloat curLen;


	s_data[tid] = (A[bid*r+tid]-x[tid])*(A[bid*r+tid]-x[tid]);

	__syncthreads();

	curLen = r;

	for (int i = (curLen+1)/2; i > 0 ; i = (curLen+1)/2 )
	{
		if(tid < i && tid+i < curLen)
		{
			s_data[tid]   += s_data[tid+i];
		}
		curLen = i;
		__syncthreads();
	}
	
	if (tid == 0)
		dis[bid] = sqrt(s_data[0]);
}

void column_wise_distance_mv(psFloat* n, psFloat* A, int r, int c,psFloat* x)
{
	_column_wise_distance_mv<<<c,r,r>>>(n, A, r, c,x);
}




__global__ void _fill(psFloat* a, unsigned int N, unsigned int inc, psFloat val)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int location = i * inc;
	
	if (location >= N)
		return;
		
	a[location] = val;
	
} 


void fill(psFloat* a, unsigned int N, unsigned int inc, psFloat val)
{
	int numGrid = (N + 512 -1)/512;
	_fill<<<numGrid,512>>>(a, N,inc,val);
}


#define BLOCK_DIM                      16
/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param pA    pitch of matrix A given in number of columns
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param pB    pitch of matrix B given in number of columns
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
__global__ void cuComputeDistanceGlobal( psFloat* A, int wA, int pA, psFloat* B, int wB, int pB, int dim,  psFloat* AB){

	// Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
	__shared__ psFloat shared_A[BLOCK_DIM][BLOCK_DIM];
	__shared__ psFloat shared_B[BLOCK_DIM][BLOCK_DIM];
    
    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;
	
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	
	// Other variables
	psFloat tmp;
    psFloat ssd = 0;
	
    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * pA;
    step_B  = BLOCK_DIM * pB;
    end_A   = begin_A + (dim-1) * pA;
    
    // Conditions
	int cond0 = (begin_A + tx < wA); // used to write in shared memory
    int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
    int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix
    
    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
        
        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/pA + ty < dim){
            shared_A[ty][tx] = (cond0)? A[a + pA * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? B[b + pB * ty + tx] : 0;
        }
        else{
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1){
            for (int k = 0; k < BLOCK_DIM; ++k){
				tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
			}
        }
        
        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1)
        AB[ (begin_A + ty) * pB + begin_B + tx ] = ssd;
}


void pairwise_squared_distance( psFloat* RQ, psFloat* ref, int wR, int pR, psFloat* query, int wQ, int pQ, int dim )
{
	dim3 g(wQ/BLOCK_DIM, wR/BLOCK_DIM, 1);
    dim3 t(BLOCK_DIM, BLOCK_DIM, 1);

    if (wQ	%	BLOCK_DIM != 0) g.x += 1;
    if (wR	%	BLOCK_DIM != 0) g.y += 1;

	cuComputeDistanceGlobal<<<g,t>>>(ref, wR, pR, query, wQ, pQ, dim,  RQ);

}


__global__ void columnwise_min_index(int * index, psFloat * A, int r, int c)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	if (tid >= r || bid >=c)
		return;

	extern __shared__ psFloat s_data[];
	extern __shared__ psFloat curLen;
	extern __shared__ psFloat s_index[];


	s_data[tid] = A[bid*c+tid];
	index[tid] = tid;

	__syncthreads();


	curLen = r;

	for (int i = (curLen+1)/2; i > 0 ; i = (curLen+1)/2 )
	{
		if(tid < i && tid+i < curLen)
		{
			if (s_data[tid] > s_data[tid+i])
			{
				s_data[tid] =  s_data[tid+i];
				s_index[tid] = s_index[tid+i];
			}
		}
		curLen = i;
		__syncthreads();
	}

	
	if (tid == 0)
		index[bid] = s_index[0];

}

void columnwise_min_index(psFloat* n, psFloat* A, int r, int c)
{
	columnwise_min_index<<<c,r,r>>>(n, A, r, c);
}

__global__ void _copy_indexed_columns(psFloat * X, int * index, int nX, psFloat * A, int r, int c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= r || j >= nX)
		return;

	X[j*r+i] = A[index[j]*r+i];
}

void copy_indexed_columns(psFloat * X, int * index, int nX, psFloat * A, int r, int c)
{
	dim3 dimBlock(16,32);
	dim3 dimGrid((r+dimBlock.x-1)/dimBlock.x, (c+dimBlock.y-1)/dimBlock.y);

	_copy_indexed_columns<<<dimGrid,dimBlock>>>(X,index, nX, A, r, c);
}
