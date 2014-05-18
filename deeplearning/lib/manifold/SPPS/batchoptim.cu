
#include "config.h"
#include <math.h>
#include <cublas.h>
#include <float.h>
#include "util.h"



__device__ void swap(psFloat& a, psFloat& b)
{
	psFloat c = b;
	b = a;
	a = c;
}


__device__ void shft3(psFloat&a, psFloat&b, psFloat&c, const psFloat d)
{
	a=b;
	b=c;
	c=d;
}

__device__ psFloat sign(const psFloat&a, const psFloat&b)
{
	return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

__device__ psFloat GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20,ZEPS=DBL_EPSILON*1.0e-3, CGOLD=0.3819660,EPS=1.0e-18;

__device__ int ncom, dcom;

psFloat * _Finish_mnbrak;
psFloat * _Finish_brent;
psFloat * _Finish_frprmn;

int * _Step_mnbrak;

psFloat *_temp_dcom_ncom_1;

psFloat * _u, * _fu, * _x, * _fx, * _v, *_fv, * _w, *_fw, *_a,* _b;

__global__ void mnbrak_pre_step( psFloat * finish, psFloat * ax, psFloat * bx, psFloat *cx, psFloat *fa, psFloat *fb, psFloat *fc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom) return;
	
	if (fb[i] > fa[i]) {
		swap(ax[i],bx[i]);
		swap(fb[i],fa[i]);
	}
	cx[i]=bx[i]+GOLD*(bx[i]-ax[i]);

	finish[i] = 0;
	
}


__global__ void mnbrak_step1(psFloat * finish, int * step, psFloat * u, psFloat * ax, psFloat * bx, psFloat *cx, psFloat *fa, psFloat *fb, psFloat *fc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom) return;
	
	if (fb[i] <= fc[i])
	{
		finish[i] = 1;
		return;
	}
	
	psFloat  ulim,r,q;
	
	r=(bx[i]-ax[i])*(fb[i]-fc[i]);
	q=(bx[i]-cx[i])*(fb[i]-fa[i]);
	u[i]=bx[i]-((bx[i]-cx[i])*q-(bx[i]-ax[i])*r)/ (2.0*sign(max(fabs(q-r),TINY),q-r));
	ulim=bx[i]+GLIMIT*(cx[i]-bx[i]);
	if ((bx[i]-u[i])*(u[i]-cx[i]) > 0.0) 
	{
		step[i] = 0;
	} 
	else if ((cx[i]-u[i])*(u[i]-ulim) > 0.0) 
	{
		step[i] = 1;
	}
	else if ((u[i]-ulim)*(ulim-cx[i]) >= 0.0) 
	{
		u[i] = ulim;
		step[i] = 2;
	} 
	else 
	{
		u[i]=cx[i]+GOLD*(cx[i]-bx[i]);
		step[i] = 3;
	}
}

__global__ void mnbrak_step2( psFloat * finish, int * step, psFloat * u,  psFloat * ax, psFloat * bx, psFloat *cx, psFloat *fa, psFloat *fb, psFloat *fc, psFloat * fu)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom) return;
	
	if (finish[i]) return;
	
	if (step[i] == 0)
	{
		if (fu[i] < fc[i]) {
			ax[i]=bx[i];
			bx[i]=u[i];
			fa[i]=fb[i];
			fb[i]=fu[i];
			finish[i] = 1;
			return;
		} else if (fu[i] > fb[i]) {
			cx[i]=u[i];
			fc[i]=fu[i];
			finish[i] = 1;
			return;
		}
		u[i]=cx[i]+GOLD*(cx[i]-bx[i]);
	}
	else if (step[i] == 1)
	{
		if (fu < fc) {
			shft3(bx[i],cx[i],u[i],cx[i]+GOLD*(cx[i]-bx[i]));
//			shft3(fb,fc,fu,func(u));
			step[i] = 11;
		}
	}
}

__global__ void mnbrak_step3( psFloat * finish, int * step, psFloat * u,  psFloat * ax, psFloat * bx, psFloat *cx, psFloat *fa, psFloat *fb, psFloat *fc, psFloat * fu)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom) return;
	
	if (finish[i]) return;
	
    if (step[i] == 11)
	{
		fb[i] = fc[i] = fu[i];
	}
	
	shft3(ax[i],bx[i],cx[i],u[i]);
	shft3(fa[i],fb[i],fc[i],fu[i]);
	
	if (fb[i] > fc[i]) 
	{
		finish[i] = 1;
	}
	
	
}



// mnbrak利用黄金比率和二次插值确定函数的极小值点的所在区间
void batch_mnbrak(  psFloat * ax, psFloat * bx, psFloat *cx, psFloat *fa, psFloat *fb, psFloat *fc, void func(psFloat *, psFloat*))
{
	func(fa,ax);
	func(fb,bx);

	int numGrid = (ncom + 512 -1)/512;

	mnbrak_pre_step<<<numGrid,512>>>(  _Finish_mnbrak, ax,  bx, cx, fa, fb, fc);
	func(fc,cx);
	
	while (cublasFasum(ncom, _Finish_mnbrak, 1) <= ncom)
	{
		mnbrak_step1<<<numGrid,512>>>(_Finish_mnbrak, _Step_mnbrak, _u,  ax,  bx, cx, fa, fb, fc);
		func(_fu,_u);
		mnbrak_step2<<<numGrid,512>>>( _Finish_mnbrak,  _Step_mnbrak,  _u,   ax,  bx, cx, fa, fb, fc, _fu);
		func(_fu,_u);
		mnbrak_step3<<<numGrid,512>>>( _Finish_mnbrak, _Step_mnbrak,  _u,  ax,  bx, cx, fa, fb, fc,  _fu);
	}
	
	
}


__global__ void brent_pre_step(psFloat * finish,  psFloat * a, psFloat * b, psFloat * x, psFloat * w, psFloat * v,  psFloat * ax, psFloat * bx, psFloat *cx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom) return;
	
	a[i] = (ax[i] < cx[i] ? ax[i] : cx[i]);
	b[i] = (ax[i] > cx[i] ? ax[i] : cx[i]);
	
	x[i] = w[i] = v[i] = bx[i];

	finish[i] = 0;
}

__global__ void brent_step1( psFloat * finish, psFloat * xmin, psFloat * u,   psFloat * a, psFloat * b, psFloat * x, psFloat * w, psFloat * v, psFloat * fx, psFloat * fv, psFloat * fw, psFloat tol )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom || finish[i] == 1) return;
	
	psFloat xm = 0.5*(a[i] + b[i]);
	
	psFloat p,q,r, e,d, etemp;
	psFloat tol1, tol2;
		
	tol2 = 2.0*(tol1 = tol*fabs(x[i])+ZEPS);
	e = 0.0; d = 0.0;

	if (fabs(x[i]-xm) <= (tol2-0.5*(b[i]-a[i]))) {
		xmin[i] = x[i];
		finish[i] = 1;
		return;
	}
	
	if (fabs(e) > tol1) {
		r=(x[i]-w[i])*(fx[i]-fv[i]);
		q=(x[i]-v[i])*(fx[i]-fw[i]);
		p=(x[i]-v[i])*q-(x[i]-w[i])*r;
		q=2.0*(q-r);
		if (q > 0.0) p = -p;
		q=fabs(q);
		etemp=e;
		e=d;
		if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a[i]-x[i]) || p >= q*(b[i]-x[i]))
			d=CGOLD*(e=(x[i] >= xm ? a[i]-x[i] : b[i]-x[i]));
		else {
			d=p/q;
			u[i]=x[i]+d;
			if (u[i]-a[i] < tol2 || b[i]-u[i] < tol2)
				d=sign(tol1,xm-x[i]);
		}
	} else {
		d=CGOLD*(e=(x[i] >= xm ?  a[i]-x[i] : b[i]-x[i]));
	}
	u[i] = (fabs(d) >= tol1 ? x[i]+d : x[i]+sign(tol1,d));
}


__global__ void brent_step2( psFloat * finish,  psFloat * u, psFloat * a, psFloat * b, psFloat * x, psFloat * w, psFloat * v, psFloat * fx, psFloat * fv, psFloat * fw, psFloat * fu)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom || finish[i] ) return;
	
	if (fu[i] <= fx[i]) {
		if (u[i] >= x[i]) a[i]=x[i]; else b[i]=x[i];
		shft3(v[i],w[i],x[i],u[i]);
		shft3(fv[i],fw[i],fx[i],fu[i]);
	} else {
		if (u[i] < x[i]) a[i]=u[i]; else b[i]=u[i];
		if (fu[i] <= fw[i] || w[i] == x[i]) {
			v[i]=w[i];
			w[i]=u[i];
			fv[i]=fw[i];
			fw[i]=fu[i];
		} else if (fu[i] <= fv[i] || v[i] == x[i] || v[i] == w[i]) {
			v[i]=u[i];
			fv[i]=fu[i];
		}
	}

}

// 求极小值点的一维搜索过程
void batch_brent(psFloat * fx, psFloat * xmin, psFloat * ax, psFloat * bx, psFloat * cx, void f(psFloat *, psFloat*), psFloat tol)
{
	const int ITMAX=100;
	int iter;


	//a=(ax < cx ? ax : cx);
	//b=(ax > cx ? ax : cx);
	//x=w=v=bx;

	int numGrid = (ncom + 512 -1)/512;

	brent_pre_step<<<numGrid, 512>>>( _Finish_brent,_a,  _b,  _x,  _w,  _v,  ax,  bx, cx);
	f(fx, _x);
	cublasFcopy(ncom,_fx, 1,_fv,1);
	cublasFcopy(ncom,_fx, 1,_fw,1);
	for (iter=0;iter<ITMAX && cublasFasum(ncom,_Finish_brent,1) < ncom - 0.1 ;iter++) {
		brent_step1<<<numGrid,512>>>( _Finish_brent,  xmin,  _u,   _a,  _b, _x,  _w,  _v,  fx,  _fv,  _fw, tol );
		f(_fu,_u);
		brent_step2<<<numGrid,512>>>( _Finish_brent, _u, _a,  _b,  _x,  _w,  _v,  fx,  _fv,  _fw, _fu  );
	}
//	nrerror("Too many iterations in brent");
	cublasFcopy(ncom,_x, 1,xmin,1);
}

/**
* 
*/


void (*nrfunc)(psFloat*, psFloat*);
psFloat * xt;
psFloat * pcom_p;
psFloat * xicom_p;

void f1dim(psFloat * fx,  psFloat * x)
{
	int j;

	psFloat* pcom = pcom_p, * xicom = xicom_p;

	cublasFcopy(ncom*dcom,pcom, 1,xt,1);
//	cublasFaxpy(ncom,x,xicom, 1,xt,1);
	column_wise_scal(_temp_dcom_ncom_1, 1, xicom , dcom, ncom, x);
	cublasFaxpy(ncom*dcom,1,_temp_dcom_ncom_1, 1,xt,1);
	//for (j=0;j<ncom;j++)
	//	xt[j]=pcom[j]+x*xicom[j];
	nrfunc(fx, xt);
}


psFloat * _ax, * _bx, * _xx , * _fxx, * _fa, * _fb, * _xmin;
void batch_linmin(psFloat* p, psFloat* xi, psFloat* fret, void func(psFloat *, psFloat*))
{
	int j;
	const psFloat TOL=1.0e-8;

	nrfunc=func;

	cublasFcopy(ncom*dcom,p, 1,pcom_p,1);
	cublasFcopy(ncom*dcom,xi, 1,xicom_p,1);
	//Vec_DP &pcom=*pcom_p,&xicom=*xicom_p;
	//for (j=0;j<n;j++) {
	//	pcom[j]=p[j];
	//	xicom[j]=xi[j];
	//}
	
	cudaMemset(_ax,0,ncom*sizeof(psFloat));
	fill(_xx,ncom,1,1.0);

	batch_mnbrak( _ax,_xx,_bx,_fa,_fxx,_fb,f1dim);
	batch_brent(fret, _xmin, _ax,_xx,_bx,f1dim,TOL);

//	cublasFscal(ncom, xmin,xi,1);
//	cublasFaxpy(ncom,1,xi,1,p,1);
	//cublasFcopy(ncom*dcom,xim, 1,xi,1);
	column_wise_scal(xi, 1, xi , dcom, ncom, _xmin);
	cublasFaxpy(ncom*dcom,1,p, 1,xi,1);
	//for (j=0;j<n;j++) {
	//	xi[j] *= xmin;
	//	p[j] += xi[j];
	//}
	//delete xicom_p;
	//delete pcom_p;
}

/**
* 共轭梯度法进行最优化。
*/

__global__ void batch_frprmn_step1( psFloat * finish, psFloat * fret, psFloat * fp, psFloat ftol)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ( i >= ncom || finish[i] == 1) return;
	
	if (2.0*fabs(fret[i]-fp[i]) <= ftol*(fabs(fret[i])+fabs(fp[i])+EPS))
		finish[i] = 1;

}



__global__ void batch_frprmn_step2( psFloat * finish, psFloat * gg, psFloat * dgg, psFloat * xi, psFloat * g)
{
	int bid = blockIdx.x ;
	int tid = threadIdx.x;

	// Add the overflow checking!
	
	if (tid >= dcom || bid >= ncom || finish[bid] == 1 )
		return;
		
	extern __shared__ psFloat gg_data[]; 
	extern __shared__ psFloat dgg_data[]; 
	
	gg_data[tid] = g[bid*dcom+tid]*g[bid*dcom+tid];
	dgg_data[tid] = (xi[bid*dcom+tid]+g[bid*dcom+tid])*xi[bid*dcom+tid];

	__syncthreads();

	for (int i = (dcom+1)/2; i > 0 ; i /=2)
	{
		if(tid < i && tid+i < dcom)
		{
			gg_data[tid] = gg_data[tid] + gg_data[tid+i];
			dgg_data[tid] = dgg_data[tid] + dgg_data[tid+i];
		}
		__syncthreads();
	}
	
	if (tid == 0)
	{
		gg[bid] = gg_data[0];
		dgg[bid] = dgg_data[0];
		
		if (dgg[bid] == 0.0)
		{
			finish[bid] = 1;
			return;
		}	
		
	}
}

__global__ void batch_frprmn_step3( psFloat * finish, psFloat * xi, psFloat * g, psFloat * h, psFloat * gg, psFloat * dgg )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= dcom || j >= ncom)
		return;
		
	psFloat gam = dgg[j]/gg[j];

	g[j*dcom+i] = -xi[j*dcom+i];
	xi[j*dcom+i] = h[j*dcom+i] = g[j*dcom+i] + gam*h[j*dcom+i];
}



psFloat * _gg, * _fp, * _dgg, * _g, * _xi, * _h;

void batch_frprmn(int n, int d, psFloat* p, const psFloat ftol, int &iter, psFloat* fret,
	void func(psFloat *, psFloat* ), void dfunc(psFloat*, psFloat*))
{
	const int ITMAX=200;
	const psFloat EPS=1.0e-18;
	int j,its;

	func(_fp,p);
	dfunc(p,_xi);

	ncom = n;
	dcom = d;

	cublasFscal(ncom*dcom,-1,_xi,1);
	cublasFcopy(ncom*dcom,_xi, 1,_g,1); 
	cublasFcopy(ncom*dcom,_g, 1,_h,1);
	cublasFcopy(ncom*dcom,_g, 1,_xi,1);

	//for (j=0;j<n;j++) {
	//	g[j] = -xi[j];
	//	xi[j]=h[j]=g[j];
	//}

	int numGrid = (ncom + 512 -1)/512;


	cudaMemset(_Finish_frprmn,0,ncom*sizeof(psFloat));

	for (its=0;its<ITMAX && cublasFasum(ncom, _Finish_frprmn,1) < ncom ;its++) {
		iter=its;
		batch_linmin(p,_xi,fret,func);
		batch_frprmn_step1<<<numGrid,512>>>( _Finish_frprmn, fret, _fp, ftol);
		if (cublasFasum(ncom,_Finish_frprmn,1) >=  ncom - 0.1)
			return;
		
		cublasFcopy(ncom,fret, 1,_fp,1); //fp=fret;
		dfunc(p,_xi);
		batch_frprmn_step2<<<numGrid,512>>>(_Finish_frprmn, _gg, _dgg, _xi,  _g);

/*		column_wise_norm2(gg, g, dcom,  ncom);
		
		dgg=gg=0.0;
		gg = cublasFnrm2(ncom, g,1);
		gg *= gg;
		dgg = cublasFdot(ncom, xi,1,xi,1) + cublasFdot(ncom, g,1,xi,1);
*/
//		for (j=0;j<n;j++) {
//			gg += g[j]*g[j];
////		  dgg += xi[j]*xi[j];
//			dgg += (xi[j]+g[j])*xi[j];
//		}

		batch_frprmn_step3<<<numGrid,512>>>(_Finish_frprmn, _xi, _g, _h, _gg, _dgg);
/*		if (gg == 0.0)
			return;
		gam=dgg/gg;


		cublasFscal(ncom*dcom,-1,xi,1);
		cublasFcopy(ncom*dcom,xi, 1,g,1);
		cublasFscal(ncom,gam,h,1);
		cublasFaxpy(ncom,1,g, 1,h,1);
		cublasFcopy(ncom,h, 1,xi,1);
*/
		//for (j=0;j<n;j++) {
		//	g[j] = -xi[j];
		//	xi[j]=h[j]=g[j]+gam*h[j];
		//}
	}

}


void batch_optim_init(int maxdim, int maxn)
{

	cublasAlloc(maxn, sizeof(psFloat), (void**)& _Finish_mnbrak);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _Finish_brent);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _Finish_frprmn);

	cublasAlloc(maxn, sizeof(int), (void**)& _Step_mnbrak);


	cublasAlloc(maxn, sizeof(psFloat), (void**)& _u);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fu);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _x);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fx);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _v);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fv);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _w);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fw);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _a);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _b);

	cublasAlloc(maxn, sizeof(psFloat), (void**)& _ax);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fa);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _bx);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fb);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _xx);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fxx);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _xmin);

	cublasAlloc(maxdim*maxn, sizeof(psFloat),(void **) & _temp_dcom_ncom_1);

	cublasAlloc(maxn, sizeof(psFloat), (void**)& _gg);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _fp);
	cublasAlloc(maxn, sizeof(psFloat), (void**)& _dgg);

	cublasAlloc(maxdim*maxn, sizeof(psFloat),(void **) & _g);
	cublasAlloc(maxdim*maxn, sizeof(psFloat),(void **) & _h);
	cublasAlloc(maxdim*maxn, sizeof(psFloat),(void **) & _xi);


	cublasAlloc(maxdim*maxn, sizeof(psFloat),(void **) & xt);
	cublasAlloc(maxdim*maxn, sizeof(psFloat),(void **) & pcom_p);
	cublasAlloc(maxdim*maxn, sizeof(psFloat),(void **) & xicom_p);

}

void batch_optim_final()
{

	cublasFree( _Finish_mnbrak);
	cublasFree( _Finish_brent);
	cublasFree( _Finish_frprmn);

	cublasFree( _Step_mnbrak);


	cublasFree( _u);
	cublasFree( _fu);
	cublasFree( _x);
	cublasFree( _fx);
	cublasFree( _v);
	cublasFree( _fv);
	cublasFree( _w);
	cublasFree( _fw);
	cublasFree( _a);
	cublasFree( _b);

	cublasFree( _ax);
	cublasFree( _fa);
	cublasFree( _bx);
	cublasFree( _fb);
	cublasFree( _xx);
	cublasFree( _fxx);
	cublasFree( _xmin);

	cublasFree( _temp_dcom_ncom_1);

	cublasFree( _gg);
	cublasFree( _fp);
	cublasFree( _dgg);

	cublasFree(_g);
	cublasFree(_h);
	cublasFree(_xi);

	cublasFree(xt);
	cublasFree(pcom_p);
	cublasFree(xicom_p);

}