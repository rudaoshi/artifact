
#include "config.h"
#include "spps.h"

#include <math.h>
#include <cublas.h>
#include <cutil.h>


#include "util.h"
#include "cumath.h"
#include "optim.h"


psFloat* x;
psFloat* diffVecs;
psFloat* normDiffV;
psFloat* kernelVal;
psFloat* kernelD;
psFloat* kernelD2;

psFloat* b;
psFloat* Jbx;
psFloat* Jbs;
psFloat* p;
psFloat* Jpx;
psFloat* Jps;

psFloat* q;

psFloat* Hbxx;
psFloat* Hbxs;

psFloat* mx;
psFloat* Jmx;
psFloat* Jms;

psFloat* y;
psFloat dis;
psFloat* Jdx;
psFloat* Hdxx;
psFloat* Jds;
psFloat* Hdxs;

psFloat* gx;     // 求投影坐标时，对x进行共轭梯度优化时使用的g
psFloat* hx;     // 求投影坐标时，对x进行共轭梯度优化时使用的h


psFloat* ones_M;  // M维全1向量
psFloat* ones_d;  // d维全1向量
//Finished: 必须消除使用MxM矩阵
//psFloat* tempMxM; // 较大的矩阵，至少是M x M的  
psFloat* tempD;
psFloat* tempd;
psFloat* tempM;


psFloat * mX;
psFloat * projY;


SPPS s;


void display(char * msg, psFloat * dm, int r , int c)
{
	static psFloat temp[1024*1024];
	
	for (int i = 0;i < r; i++)
	{
		for (int j = 0;j < c; j++)
		{
			temp[j*r+i]  = -1;
		}
	}
	
	cublasGetMatrix(r,c, sizeof(psFloat), dm, r, temp,r);
	
	printf(msg); printf("\n[");
	 
	for (int i = 0;i < r; i++)
	{
		for (int j = 0;j < c; j++)
		{
			printf("%f ", temp[j*r+i]);
		}
		
		if (i < r-1)
			printf("\n");
	}
	printf("]\n");
}

void SPPS_init(psFloat* Y, int D, int N, psFloat* initS, psFloat* initT, int d, int M, KernelType kernel, psFloat* Sigma, int * nngraph, int neighbor_pair)
{
	if (cublasInit() != CUBLAS_STATUS_SUCCESS)
		return;

	// Fill the elements of S

	s.D = D;
	s.N = N;
	s.d = d;
	s.M = M;
	s.kernel = kernel;
	s.neighbor_pair = neighbor_pair;

	cublasAlloc(s.D*s.N, sizeof(psFloat), (void **) &s.Y);
	cublasSetMatrix(s.D, s.N, sizeof(psFloat), Y,D, s.Y, s.D);
	
//	display("Computing the SPPS for data set:", s.Y , s.D , s.N);
	
	cublasAlloc(s.d*s.N, sizeof(psFloat), (void **) &s.X);

	cublasAlloc(s.D*s.M, sizeof(psFloat), (void **) &s.T);
	cublasSetMatrix(s.D, s.M, sizeof(psFloat),  initT,D,s.T, s.D);


	cublasAlloc(s.d*s.M, sizeof(psFloat),(void **) &s.S);
	cublasSetMatrix(s.d, s.M, sizeof(psFloat), initS,d,  s.S, s.d);

	
	cublasAlloc(s.M, sizeof(psFloat),(void **) &s.Sigma);
	cublasSetVector( s.M, sizeof(psFloat), Sigma, 1, s.Sigma,1);

	cublasAlloc(2*s.neighbor_pair, sizeof(int), (void **) &s.nngraph );
	cublasSetMatrix( s.neighbor_pair,2, sizeof(int), nngraph, s.neighbor_pair ,  s.nngraph, s.neighbor_pair);

	// malloc the internal temporary memories

//	cublasAlloc(s.d, sizeof(psFloat),(void **) & x);

	cublasAlloc(s.d*s.M, sizeof(psFloat),(void **) & diffVecs);
	cublasAlloc(s.M, sizeof(psFloat),(void **) & normDiffV);
	cublasAlloc(s.M, sizeof(psFloat),(void **) & kernelVal);
	cublasAlloc(s.M, sizeof(psFloat),(void **) & kernelD);
	cublasAlloc(s.M, sizeof(psFloat),(void **) & kernelD2);


	cublasAlloc(s.M, sizeof(psFloat),(void **) & b);
	cublasAlloc(s.d*s.M, sizeof(psFloat),(void **) & Jbx);
	cublasAlloc(s.d*s.M, sizeof(psFloat),(void **) & Jbs);
	cublasAlloc(s.M, sizeof(psFloat),(void **) & p);
	cublasAlloc(s.d*s.M, sizeof(psFloat),(void **) & Jpx);
	cublasAlloc(s.d*s.M, sizeof(psFloat),(void **) & Jps);
	cublasAlloc(s.M, sizeof(psFloat),(void **) & q);

	cublasAlloc(s.d*s.d, sizeof(psFloat),(void **) & Hbxx);
	cublasAlloc(s.d*s.d, sizeof(psFloat),(void **) & Hbxs);

	cublasAlloc(s.D, sizeof(psFloat),(void **) & mx);
	cublasAlloc(s.d*s.D, sizeof(psFloat),(void **) & Jmx);
	cublasAlloc(s.d*s.D, sizeof(psFloat),(void **) & Jms);

//	cublasAlloc(s.D, sizeof(psFloat),(void **) & y);
//	cublasAlloc(s.d, sizeof(psFloat),(void **) & Jdx);
	cublasAlloc(s.d*s.d, sizeof(psFloat),(void **) & Hdxx);
	cublasAlloc(s.d, sizeof(psFloat),(void **) & Jds);
	cublasAlloc(s.d*s.d, sizeof(psFloat),(void **) & Hdxs);

	cublasAlloc(s.d, sizeof(psFloat),(void **) & gx);
	cublasAlloc(s.d, sizeof(psFloat),(void **) & hx);

	// malloc the external temporary memories

	cublasAlloc(s.M, sizeof(psFloat),(void **) & ones_M);
	fill(ones_M, s.M,1, 1.0);

	cublasAlloc(s.d, sizeof(psFloat), (void **) & ones_d);
	fill(ones_d, s.d,1, 1.0);

	cublasAlloc(s.D, sizeof(psFloat),(void **) & tempD);
	cublasAlloc(s.d, sizeof(psFloat),(void **) & tempd);
	cublasAlloc(s.M, sizeof(psFloat),(void **) & tempM);

	cublasAlloc(s.D*s.M, sizeof(psFloat), (void **) &mX);
	cublasAlloc(s.D*s.M, sizeof(psFloat), (void **) &projY);
	
	optim_init(max(s.D,s.M));

}


void SPPS_final()
{

	// Free the elements of SPPS
	cublasFree(s.Y);

	cublasFree(s.T);

	cublasFree(s.S);
	
	cublasFree(s.Sigma);

	cublasFree(s.nngraph );

	// free the internal temporary memories

//	cublasFree( x);
	cublasFree( diffVecs);
	cublasFree( normDiffV);
	cublasFree( kernelVal);
	cublasFree( kernelD);
	cublasFree( kernelD2);


	cublasFree( b);
	cublasFree( Jbx);
	cublasFree( Jbs);
	cublasFree( p);
	cublasFree( Jpx);
	cublasFree( Jps);
	cublasFree( q);

	cublasFree( Hbxx);
	cublasFree( Hbxs);

	cublasFree( mx);
	cublasFree( Jmx);
	cublasFree( Jms);

//	cublasFree( y);
	cublasFree( Jdx);
	cublasFree( Hdxx);
	cublasFree( Jds);
	cublasFree( Hdxs);

	cublasFree( gx);
	cublasFree( hx);

	// free the external temporary memories

	cublasFree( ones_M);
	cublasFree( ones_d);

	cublasFree(  tempD);
	cublasFree(  tempd);
	cublasFree(  tempM);
	cublasFree(  mX);
	cublasFree(  projY);

	
	optim_final();


}

void slps_regkernel( CalculationType calType)
{

	column_wise_norm2(normDiffV, diffVecs, s.d, s.M);
	
#ifdef _DEBUG_DETAIL
	display("the resulting normDiffV matrix is:", normDiffV, 1 , s.M);
#endif

	if (s.kernel == Gaussian)
	{
		gausian_kernel( kernelVal, kernelD, kernelD2, normDiffV, s.M, s.Sigma, calType);

	}
	else if(s.kernel == Quadratic)
	{
		quadratic_kernel( kernelVal, kernelD, kernelD2, normDiffV, s.M, s.Sigma, calType);

	}

#ifdef _DEBUG_DETAIL
	display("the resulting kernelVal matrix is:", kernelVal, 1 , s.M);
#endif
	
}




void slps_kernelregbasis(psFloat * _x, CalculationType calType)
{
//	psFloat* x = x;

	// calculate the diffVecs
#ifdef _DEBUG_DETAIL
	display("the S matrix is:", s.S, s.d , s.M);
#endif
	column_wise_add(diffVecs, 1,  s.S, s.d, s.M, -1, _x);
	//cublasFcopy(s.d * s.M, s.S,  1, diffVecs, 1);
	//cublasFger (s.d, s.M, 1.0, x, 1, ones_M, 1, diffVecs, s.d);

#ifdef _DEBUG_DETAIL
	display("the resulting diffVecs matrix is:", diffVecs, s.d , s.M);
#endif
	
	
	// calculate the kernel value and kernel differences
	slps_regkernel(calType);
	
	// calculate b
	psFloat sumval = cublasFasum(s.M, kernelVal, 1);
	
	if (sumval < 1e-6)
	{ 
		sumval = 1;
	}
	
	cublasFcopy (s.M, kernelVal,  1, b, 1);
	cublasFscal (s.M, 1/sumval, b, 1);
	
#ifdef _DEBUG_DETAIL
	display("the resulting b matrix is:", b, 1 , s.M);
#endif
	
	if (calType == Eval)
		return;
			
	// Calculate p
	cublasFcopy (s.M, kernelD,  1, p, 1);
	cublasFscal (s.M, -2/sumval, p, 1);
#ifdef _DEBUG_DETAIL
	display("the resulting p matrix is:", p, 1 , s.M); 
#endif
	// Calculate q
	cublasFcopy (s.M, kernelD2,  1, q, 1);
	cublasFscal (s.M, 4/sumval, q, 1);
#ifdef _DEBUG_DETAIL
	display("the resulting q matrix is:", q, 1 , s.M); 
#endif
	// Calculate Jbx
	// Jbx = diffVecs*(diag(p)-p*b');
	// 1. Jbx = diffVecs*diag(p);
	column_wise_scal(Jbx,diffVecs, s.d, s.M, p);
	// 2. tempd = diffVecs*p;
	cublasFgemv('N', s.d,s.M, 1, diffVecs, s.d, p, 1, 0, tempd,1);
	// 3. Jbx = Jbx - tempd*b'
	cublasFger(s.d,s.M, -1, tempd, 1, b, 1, Jbx, s.d);
#ifdef _DEBUG_DETAIL
	display("the resulting Jbx matrix is:", Jbx, s.d , s.M);
#endif
			
	// Calculate Jpx
	// Jpx =  diffVecs*(diag(q) - p*p');
	// 1. Jpx = diffVecs*diag(q);
	column_wise_scal(Jpx,diffVecs, s.d, s.M, q);
	// 2. tempd = diffVecs*p; has already been calculated.
	// cublasFgemv('N', s.d,s.M, 1, diffVecs, s.d, p, 1, 0, tempd,1);
	// 3. Jpx = Jpx - tempd*p'
	cublasFger(s.d,s.M, -1, tempd, 1, p, 1, Jpx, s.d);

}

void slps_map(psFloat * _mx, psFloat * _Jmx, psFloat * _x,  CalculationType calType)
{

	slps_kernelregbasis(_x, calType);
	// Compute mx
	// ps.mx = ps.T*ps.b;
	cublasFgemv('N', s.D,s.M,1,  s.T, s.D, 
			b, 1, 0, _mx, 1);
			
			
	if (calType == Eval )
		return;
	
	// Compute Jmx
	// ps.Jmx = ps.Jbx*ps.T';
	
	cublasFgemm('N','T',s.d,s.D,s.M, 1, Jbx, s.d, 
			s.T, s.D, 0, _Jmx, s.d);
	
}

// Compute the distance 
void slps_dist(psFloat * dist, psFloat * _mx, psFloat * _Jmx, psFloat * _Jdx, psFloat * _Hdxx, psFloat * _x, psFloat * _y, CalculationType calType)
{
	slps_map( _mx, _Jmx, _x, calType);
	
	// tempD = y - ps.mx;
	cublasFcopy(s.D, _y, 1,  tempD, 1);
	cublasFaxpy(s.D, -1, _mx, 1, tempD, 1);.
	
	// dist = norm(tempD);
	*dist = cublasFnrm2(s.D, tempD, 1);
	*dist *= *dist;
	
	if (calType == Eval)
		return;
	
	//  Jdx = 2*ps.Jmx*(ps.mx-y);
#ifdef _DEBUG_DETAIL
	display("the resulting Jmx matrix is:", _Jmx, s.d , s.D);
	display("the resulting y-mx matrix is:", tempD, 1 , s.D);
#endif
	cublasFgemv('N', s.d, s.D, -2, _Jmx, s.d,  tempD, 1, 1, _Jdx,1);
	
	if (calType == Jacobbi)
		return;
	
	// Calculate Hdxx
	psFloat sumP = cublasFasum(s.M, p, 1);
	
	cudaMemset(_Hdxx,0,s.d*s.d*sizeof(psFloat));

	for (int i = 0; i < s.M; i++)
	{
		//Calculate Hbxx(:,:,i);
		//(b(i)*sum(p)-p(i))*eye(d) + diffVecs(:,i)*Jpx(:,i)' - diffVecs*p*Jbx(:,i)' - b(i)*diffVecs*Jpx';
		
		cudaMemset(Hbxx,0,s.d*s.d*sizeof(psFloat));
		//Hbxx = (b(i)*sum(p)-p(i))*eye(d)
		fill(Hbxx,s.d*s.d, s.d, b[i]*sumP-p[i]);
		//Hbxx += diffVecs(:,i)*Jpx(:,i)'
		cublasFger(s.d,s.d, 1, diffVecs + i*s.d, 1, Jpx + i*s.d, 1, Hbxx, s.d);
		
		//Hbxx +=  - diffVecs*p*Jbx(:,i)'
		cublasFgemv('N', s.d,s.M, 1, diffVecs, s.d, p, 1, 0, tempd,1);
		cublasFger(s.d,s.d, -1, tempD, 1, Jbx + i*s.d, 1, Hbxx, s.d);
		
		//Hbxx +=  - b(i)*diffVecs*Jpx'
		cublasFgemm('N','T',s.d,s.M,s.d, -b[i], diffVecs, s.d, 
			Jpx, s.d, 0, Hbxx, s.d);

		// Hdxx = Hdxx + 2*dot(ps.T(:,i),ps.mx - y)*ps.Hbxx(:,:,i) + 2*ps.Jbx(:,i)*ps.T(:,i)'*ps.T*ps.Jbx';
		psFloat coe = - cublasFdot(s.D, s.T+ i*s.D, 1, tempD, 1);
		// Hdxx += 2*dot(ps.T(:,i),ps.mx - y)*ps.Hbxx(:,:,i)
		cublasFaxpy(s.d*s.d, 2*coe, Hbxx, 1, _Hdxx,1);
		
		// tempM = ps.T'*ps.T(:,i)
		cublasFgemv('T', s.M,s.D, 1, s.T, s.D, s.T + i*s.D, 1, 0, tempM,1);
		// tempd = ps.Jbx*tempM = ps.Jbx*ps.T'*ps.T(:,i)
		cublasFgemv('N', s.d,s.M, 1, Jbx, s.d, tempM, 1, 0, tempd,1);
		// Hdxx += 2*ps.Jbx(:,i)*ps.T(:,i)'*ps.T*ps.Jbx' = 2*ps.Jbx(:,i)*tempd';
		cublasFger(s.d,s.d, 2, Jbx + i*s.d, 1, tempd, 1, _Hdxx, s.d);

	}	
}


// 优化辅助函数,用于求某点y到主曲面的投影指标
// 本函数用于求解某点y到S(x)的距离
// 使用前必须将y拷贝到变量y中。
psFloat slps_dist_help(psFloat* _x )
{

//	cublasFcopy(s.d, x, 1, x,1);


	slps_dist(& dis, mx, Jmx, Jdx, Hdxx, _x, y,Eval);

//	static int i = 0;
	
//	i ++;
	
//	if (i % 10 == 0)
#ifdef _DEBUG_DETAIL
    display("computing the object value at point:", _x, 1 , s.d);
#endif

	return dis;
}

// 优化辅助函数,用于求某点y到主曲面的投影指标
// 本函数用于求解某点y到S(x)的距离对x的导数
// 使用前必须将y拷贝到变量y中，并且slps_dist_help在之前紧接着被调用过。
void slps_dist_Jacobbi_help(psFloat* _x ,psFloat*  _Jdx )
{

	slps_dist(& dis, mx, Jmx, _Jdx, Hdxx, _x, y,Jacobbi) ;

#ifdef _DEBUG_DETAIL	
	display("computing the Jacobbi matrix at point:", _x, 1 , s.d);
	display("the resulting Jacobbi matrix is:", _Jdx, 1 , s.d);
#endif
	
//	cublasFcopy(s.d, Jdx, 1, Jdx,1);
}

void slps_project(psFloat * _x, psFloat* _y, psFloat* x0, bool withInitVal)
{
	/**
	* 寻找到对应的初始点x
	*/
	
//	x = x0;
	y = _y;
	
	if (!withInitVal)
	{
		column_wise_distance(tempM, s.T, s.D, s.M,y);
//		display("the pairwise distance are:", tempM, 1 , s.M);

		// 寻找tempM中最小值的位置。
		int index = cublasIfamin(s.M,tempM,1);
		index --;
		cublasFcopy(s.d, s.S + index*s.d, 1, _x,1);
	}
	else
	{
		cublasFcopy(s.d, x0, 1, _x,1);
	}
#ifdef _DEBUG_DETAIL	
	display("The initial point is:", _x, 1 , s.d);
#endif
	int iter;
	psFloat fret;
	psFloat ftol = 1e-6;
	
	// the resulting x is the projection index.
	frprmn(s.d, _x, ftol, iter, fret, &slps_dist_help,&slps_dist_Jacobbi_help);

}

void slps_map(psFloat * Y, psFloat * _X,  int N)
{
	for ( int i = 0; i <  N; i++)
	{
#ifdef _DEBUG_DETAIL	
		display("Calculate the mapping for the point:",  _X + s.d*i, 1 , s.d);
#endif
		slps_map( Y + s.D*i, Jmx, _X + s.d*i, Eval);
		
	}
}

void slps_project(psFloat * _X, psFloat * Y, int N)
{
	for ( int i = 0; i < N; i++)
	{
//		printf("Computing the projection of %d-th sample\n", i);
		//if (i == 100)
		//{
		//	bool stop;
		//	stop = true;
		//}
		slps_project(  _X + s.d*i, Y + s.D*i,0 , false);
		
	}
}


void slps_map_train(psFloat * _mX)
{
	for ( int i = 0; i < s.M; i++)
	{
#ifdef _DEBUG_DETAIL	
		display("Calculate the mapping for the point:",  s.S + s.d*i, 1 , s.d);
#endif
		slps_map( mX + s.D*i, Jmx,s.S + s.d*i, Eval);
	}

	cublasGetMatrix(s.D, s.M, sizeof(psFloat), mX, s.D, _mX, s.D);
}


void slps_project_train(psFloat * _X )
{
	for ( int i = 0; i < s.N; i++)
	{
//		printf("Computing the projection of %d-th sample\n", i);
		//if (i == 100)
		//{
		//	bool stop;
		//	stop = true;
		//}
		slps_project(projY + s.d*i,  s.Y + s.D*i,0 , false);
		
	}
	cublasGetMatrix(s.d, s.N, sizeof(psFloat), projY, s.d, _X, s.d);
}


//
//void slps_get_image(psFloat * image)
//{
//	cublasGetMatrix(s.D, s.N, sizeof(psFloat), s.X, s.d, trainfeature, s.d);
//}
//
//void slps_get_trainfeature(psFloat * trainfeature)
//{
//	cublasGetMatrix(s.d, s.N, sizeof(psFloat), s.X, s.d, trainfeature, s.d);
//}