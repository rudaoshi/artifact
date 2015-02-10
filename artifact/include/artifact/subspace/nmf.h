#ifndef NMF_H
#define NMF_H

#include <liblearning/core/config.h>
#include <liblearning/core/dataset.h>


namespace subspace
{
	using namespace core;
	using namespace std;

	//% NMF by alternative non-negative least squares using projected gradients
	//% Author: Chih-Jen Lin, National Taiwan University
	//
	//% problem: 
	//% min: 1/2 ||WH-V||^2 + 1/2\lambda*||W||^2
	//%  st:  0<= H_ij<=1,  i\in {1,...,d-1}, j \in {1,...,n}
	//%           H_dj = 1, j \in {1,...,n}
	//
	//% W,H: output solution
	//% Winit,Hinit: initial solution
	//% tol: tolerance for a relative stopping condition
	//% timelimit, maxiter: limit of time and iterations
	//
	//% Reference:
	//% 1. Lin C. Projected gradient methods for nonnegative matrix factorization. Neural computation. 2007;19(10):2756-79. Available at: http://www.ncbi.nlm.nih.gov/pubmed/17716011.



	class nmf
	{
		const MatrixType & V;
		const MatrixType VT;

		MatrixType gradW;
		MatrixType gradH;

		MatrixType W;
		MatrixType H;

		MatrixType WtV; 
		MatrixType WtW;

		MatrixType HtVt; 
		MatrixType HtH;

		function<void (MatrixType &)> PWt;
		function<void (MatrixType &)> PH;

		function<void (MatrixType &, const MatrixType &)> GPWt;
		function<void (MatrixType &, const MatrixType &)> GPH;

		static int nlssubprob(MatrixType & H, MatrixType & gradH, const MatrixType & V,const MatrixType & W, 
			function<void (MatrixType &)> PH,function<void (MatrixType &, const MatrixType &)> GPH,
			NumericType lambda,NumericType tol,int maxiter);
	public:
		nmf(const MatrixType & V_);
		~nmf(void);

		void set_constraint_Wt(const function<void (MatrixType &)> & PWt_,function<void (MatrixType &, const MatrixType &)> GPWt_);
		void set_constraint_H(const function<void (MatrixType &)> & PH_,function<void (MatrixType &, const MatrixType &)> GPH_);

		void factorize(const MatrixType & Winit,const MatrixType & Hinit,NumericType lambdaW, NumericType lambdaH, NumericType tol, int maxiter);

	};
}

#endif
