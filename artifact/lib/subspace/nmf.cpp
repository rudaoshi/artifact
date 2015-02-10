#include <liblearning/subspace/nmf.h>

#include <algorithm>

using namespace std;
using namespace subspace;

#include <liblearning/core/platform.h>
using namespace core;

#include <boost/lexical_cast.hpp>


nmf::nmf(const MatrixType & V_):V(V_),VT(V_.transpose()),PWt(0),PH(0),GPWt(0),GPH(0)
{

}


nmf::~nmf(void)
{
}

void nmf::set_constraint_Wt(const function<void (MatrixType &)> & PWt_,function<void (MatrixType &, const MatrixType &)> GPWt_)
{
	PWt = PWt_;
	GPWt = GPWt_;
}

void nmf::set_constraint_H(const function<void (MatrixType &)> & PH_,function<void (MatrixType &, const MatrixType &)> GPH_)
{
	PH = PH_;
	GPH = GPH_;
}

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


void nmf::factorize(const MatrixType & Winit,const MatrixType & Hinit,NumericType lambdaW, NumericType lambdaH, NumericType tol, int maxiter)
{
  auto & log = platform::Instance().get_log(Debug,"nmf");
	W = Winit;
	log.info("W = Winit;");
	H = Hinit;
	log.info("H = Hinit;");

	log.info("size of W = [" + boost::lexical_cast<string>(W.rows()) + "," + boost::lexical_cast<string>(W.cols()) + "]" );
	log.info("size of H = [" + boost::lexical_cast<string>(H.rows()) + "," + boost::lexical_cast<string>(H.cols()) + "]" );
	log.info("size of V = [" + boost::lexical_cast<string>(V.rows()) + "," + boost::lexical_cast<string>(V.cols()) + "]" );
	log.info("size of VT = [" + boost::lexical_cast<string>(VT.rows()) + "," + boost::lexical_cast<string>(VT.cols()) + "]" );

	gradW = W*(H*H.transpose()) - V*H.transpose();
	if (lambdaW > 0) gradW.noalias() += lambdaW*W; 
	log.info("gradW calculated");
	gradH = (W.transpose()*W)*H - W.transpose()*V; //  eq (2) in 1;
	if (lambdaH > 0) gradH.noalias() += lambdaH*H;
	log.info("gradH calculated");

	NumericType initgrad = sqrt(gradW.squaredNorm() + gradH.squaredNorm());//norm([gradW; gradH'],'fro');
	log.info("Init gradient norm " + boost::lexical_cast<string>(initgrad));

	NumericType tolW = max(NumericType(0.001),tol)*initgrad; 
	NumericType tolH = tolW;

	int iter; 
	NumericType projnorm;
	for (iter=0; iter< maxiter;iter ++)
	{
		// stopping condition
		//NumericType norm_gradH = 0;
		//for (int i = 0; i< gradH.size();i++)
		//{
		//	if ((gradH(i) >= 0 && H(i) <= 0.01) || gradH(i) <=0 && H(i) >= 0.99)
		//		continue;
		//	norm_gradH += gradH(i)*gradH(i);
		//}
		if (GPH)
			GPH(gradH,H);
		if (GPWt)
			GPWt(gradW,W);
		projnorm = sqrt(gradW.squaredNorm()+ gradH.squaredNorm());
		if (projnorm < tol*initgrad )
			break;
		
		log.info("Starting Optimizing W");
		W.transposeInPlace();
		gradW.transposeInPlace();
		int iterW = nlssubprob(W,gradW,VT,H.transpose(),PWt,GPWt,lambdaW,tolW,5);
		W.transposeInPlace();
		gradW.transposeInPlace();
		log.info("Finishing Optimizing W");
		if (iterW==0)
		{	
			tolW = 0.1 * tolW;
		}
		log.info("Starting Optimizing H");
		int iterH = nlssubprob(H,gradH,V,W,PH,GPH,lambdaW,tolH,5);
		log.info("Finishing Optimizing H");
		if (iterH==0)
		{
			tolH = 0.1 * tolH;
		}
		
    
		if (iter%10==0)
			log.info("."); 
	}
	log.info("Iter = " + boost::lexical_cast<string>(iter) + " Final proj-grad norm " + boost::lexical_cast<string>(projnorm));

}



int nmf::nlssubprob(MatrixType & H, MatrixType & gradH, const MatrixType & V,const MatrixType & W, 
	function<void (MatrixType &)> PH,function<void (MatrixType &, const MatrixType &)> GPH,
	NumericType lambda,NumericType tol,int maxiter)
{
//% H, grad: output solution and gradient
//% iter: #iterations used
//% V, W: constant matrices
//% Hinit: initial solution
//% tol: stopping tolerance
//% maxiter: limit of iterations
auto & log = platform::Instance().get_log(Debug,"nmf");

	MatrixType WtV = W.transpose()*V; 
	MatrixType WtW = W.transpose()*W;

	NumericType alpha = 1; 
	NumericType beta = 0.1;

	int iter;
	log.info("nlssubprob: Begin Optimization");
	for(iter = 0;iter<maxiter;iter++)
	{
		gradH = WtW*H - WtV + lambda*H;
		if (GPH)
			GPH(gradH,H);
		NumericType projgrad = gradH.norm();// % norm(grad(grad < 0 | H >0));
		if (projgrad < tol)
			break;
		
		MatrixType Hp;
		bool decr_alpha;
		// search step size
		log.info("nlssubprob: Begin Inner Search");
		for (int inner_iter=0;inner_iter<20;inner_iter++)
		{
			// Hn = max(H - alpha*grad, 0); d = Hn-H;
			MatrixType Hn = H - alpha*gradH;
			if (PH)
				PH(Hn);

			MatrixType d = Hn - H;

			NumericType gradd= (gradH.array()*d.array()).sum();
			NumericType dQd = ((WtW*d).array()*d.array()).sum();
			
			bool suff_decr = 0.99*gradd + 0.5*dQd < 0;  // % eq (17) in Ref. 1
			if (inner_iter==0)
			{
				decr_alpha = !suff_decr; 
				Hp = H;
			}
			
			if (decr_alpha)
			{
				if (suff_decr)
				{
					H = Hn; 
					break;
				}
				else
				{
					alpha = alpha * beta;
				}
			}
			else
			{
				if (!suff_decr || (Hp-Hn).squaredNorm() == 0.0)
				{
					H = Hp; break;
				}
				else
				{
					alpha = alpha/beta; Hp = Hn;
				}
			}
			log.info("nlssubprob: Finishing Inner Search: " + boost::lexical_cast<string>(inner_iter));

		}
	}
	log.info("nlssubprob: Finishing Optimization");
	if (iter==maxiter)
	{
		log.info("Max iter in nlssubprob");
	}
	return iter;
}

