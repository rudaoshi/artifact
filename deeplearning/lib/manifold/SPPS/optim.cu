
#include "config.h"
#include <math.h>
#include <cublas.h>
#include <float.h>



void swap(psFloat& a, psFloat& b)
{
	psFloat c = b;
	b = a;
	a = c;
}


void shft3(psFloat&a, psFloat&b, psFloat&c, const psFloat d)
{
	a=b;
	b=c;
	c=d;
}

psFloat sign(const psFloat&a, const psFloat&b)
{
	return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}


void mnbrak(psFloat &ax, psFloat &bx, psFloat &cx, psFloat &fa, psFloat &fb, psFloat &fc,
	psFloat func(const psFloat))
{
	const psFloat GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20;
	psFloat ulim,u,r,q,fu;

	fa=func(ax);
	fb=func(bx);
	if (fb > fa) {
		swap(ax,bx);
		swap(fb,fa);
	}
	cx=bx+GOLD*(bx-ax);
	fc=func(cx);
	while (fb > fc) {
		r=(bx-ax)*(fb-fc);
		q=(bx-cx)*(fb-fa);
		u=bx-((bx-cx)*q-(bx-ax)*r)/
			(2.0*sign(max(fabs(q-r),TINY),q-r));
		ulim=bx+GLIMIT*(cx-bx);
		if ((bx-u)*(u-cx) > 0.0) {
			fu=func(u);
			if (fu < fc) {
				ax=bx;
				bx=u;
				fa=fb;
				fb=fu;
				return;
			} else if (fu > fb) {
				cx=u;
				fc=fu;
				return;
			}
			u=cx+GOLD*(cx-bx);
			fu=func(u);
		} else if ((cx-u)*(u-ulim) > 0.0) {
			fu=func(u);
			if (fu < fc) {
				shft3(bx,cx,u,cx+GOLD*(cx-bx));
				shft3(fb,fc,fu,func(u));
			}
		} else if ((u-ulim)*(ulim-cx) >= 0.0) {
			u=ulim;
			fu=func(u);
		} else {
			u=cx+GOLD*(cx-bx);
			fu=func(u);
		}
		shft3(ax,bx,cx,u);
		shft3(fa,fb,fc,fu);
	}
}

psFloat brent(const psFloat ax, const psFloat bx, const psFloat cx, psFloat f(const psFloat),
	const psFloat tol, psFloat&xmin)
{
	const int ITMAX=100;
	const psFloat CGOLD=0.3819660;
	const psFloat ZEPS=DBL_EPSILON*1.0e-3;
	int iter;
	psFloat a,b,d=0.0,etemp,fu,fv,fw,fx;
	psFloat p,q,r,tol1,tol2,u,v,w,x,xm;
	psFloat e=0.0;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=f(x);
	for (iter=0;iter<ITMAX;iter++) {
		xm=0.5*(a+b);
		tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
			xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {
			r=(x-w)*(fx-fv);
			q=(x-v)*(fx-fw);
			p=(x-v)*q-(x-w)*r;
			q=2.0*(q-r);
			if (q > 0.0) p = -p;
			q=fabs(q);
			etemp=e;
			e=d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
				d=CGOLD*(e=(x >= xm ? a-x : b-x));
			else {
				d=p/q;
				u=x+d;
				if (u-a < tol2 || b-u < tol2)
					d=sign(tol1,xm-x);
			}
		} else {
			d=CGOLD*(e=(x >= xm ? a-x : b-x));
		}
		u=(fabs(d) >= tol1 ? x+d : x+sign(tol1,d));
		fu=f(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			shft3(v,w,x,u);
			shft3(fv,fw,fx,fu);
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			} else if (fu <= fv || v == x || v == w) {
				v=u;
				fv=fu;
			}
		}
	}
//	nrerror("Too many iterations in brent");
	xmin=x;
	return fx;
}

/**
* 
*/

int ncom;
psFloat(*nrfunc)(psFloat*);

psFloat * xt;


psFloat *pcom_p;
psFloat *xicom_p;

psFloat f1dim( psFloat x)
{
	int j;

	psFloat* pcom = pcom_p, * xicom = xicom_p;

	cublasFcopy(ncom,pcom, 1,xt,1);
	cublasFaxpy(ncom,x,xicom, 1,xt,1);

	//for (j=0;j<ncom;j++)
	//	xt[j]=pcom[j]+x*xicom[j];
	return nrfunc(xt);
}



void linmin(psFloat* p, psFloat* xi, psFloat&fret, psFloat func(psFloat*))
{
	int j;
	const psFloat TOL=1.0e-8;
	psFloat xx,xmin,fx,fb,fa,bx,ax;


	nrfunc=func;

	cublasFcopy(ncom,p, 1,pcom_p,1);
	cublasFcopy(ncom,xi, 1,xicom_p,1);
	//Vec_DP &pcom=*pcom_p,&xicom=*xicom_p;
	//for (j=0;j<n;j++) {
	//	pcom[j]=p[j];
	//	xicom[j]=xi[j];
	//}
	ax=0.0;
	xx=1.0;
	mnbrak(ax,xx,bx,fa,fx,fb,f1dim);
	fret=brent(ax,xx,bx,f1dim,TOL,xmin);

	cublasFscal(ncom, xmin,xi,1);
	cublasFaxpy(ncom,1,xi,1,p,1);
	//for (j=0;j<n;j++) {
	//	xi[j] *= xmin;
	//	p[j] += xi[j];
	//}
	//delete xicom_p;
	//delete pcom_p;
}




psFloat* g;
psFloat* h;
psFloat* xi;

/**
* 共轭梯度法进行最优化。
*/

void frprmn(int n, psFloat* p, const psFloat ftol, int &iter, psFloat&fret,
	psFloat func(psFloat* ), void dfunc(psFloat*, psFloat*))
{
	const int ITMAX=200;
	const psFloat EPS=1.0e-18;
	int j,its;
	psFloat gg,gam,fp,dgg;

	fp=func(p);
	dfunc(p,xi);

	ncom = n;

	cublasFscal(ncom,-1,xi,1);
	cublasFcopy(ncom,xi, 1,g,1); 
	cublasFcopy(ncom,g, 1,h,1);
	cublasFcopy(ncom,g, 1,xi,1);

	//for (j=0;j<n;j++) {
	//	g[j] = -xi[j];
	//	xi[j]=h[j]=g[j];
	//}

	for (its=0;its<ITMAX;its++) {
		iter=its;
		linmin(p,xi,fret,func);
		if (2.0*fabs(fret-fp) <= ftol*(fabs(fret)+fabs(fp)+EPS))
			return;
		fp=fret;
		dfunc(p,xi);
		dgg=gg=0.0;
		gg = cublasFnrm2(ncom, g,1);
		gg *= gg;
		dgg = cublasFdot(ncom, xi,1,xi,1) + cublasFdot(ncom, g,1,xi,1);

//		for (j=0;j<n;j++) {
//			gg += g[j]*g[j];
////		  dgg += xi[j]*xi[j];
//			dgg += (xi[j]+g[j])*xi[j];
//		}
		if (gg == 0.0)
			return;
		gam=dgg/gg;


		cublasFscal(ncom,-1,xi,1);
		cublasFcopy(ncom,xi, 1,g,1);
		cublasFscal(ncom,gam,h,1);
		cublasFaxpy(ncom,1,g, 1,h,1);
		cublasFcopy(ncom,h, 1,xi,1);

		//for (j=0;j<n;j++) {
		//	g[j] = -xi[j];
		//	xi[j]=h[j]=g[j]+gam*h[j];
		//}
	}

}


void optim_init(int maxdim)
{

	cublasAlloc(maxdim, sizeof(psFloat),(void **) & g);
	cublasAlloc(maxdim, sizeof(psFloat),(void **) & h);
	cublasAlloc(maxdim, sizeof(psFloat),(void **) & xi);
	cublasAlloc(maxdim, sizeof(psFloat),(void **) & xt);

	cublasAlloc(maxdim, sizeof(psFloat),(void **) & pcom_p);
	cublasAlloc(maxdim, sizeof(psFloat),(void **) & xicom_p);

}

void optim_final()
{
	cublasFree(g);
	cublasFree(h);
	cublasFree(xi);
	cublasFree(xt);

	cublasFree(pcom_p);
	cublasFree(xicom_p);

}