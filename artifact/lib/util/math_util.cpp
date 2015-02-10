/*
 * math_util.cpp
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */


#include <liblearning/util/math_util.h>
//#include <mkl_blas.h>
//#include <mkl_vml_functions.h>
//
//#include <mkl_vsl.h>

#include <ctime>

#include <iostream>
#include <cstdlib>
#include <cassert>

#ifdef USE_MKL
#define MAX_VECTOR_LEN 1000*1000

NumericType ONES[MAX_VECTOR_LEN];
NumericType TEMP[MAX_VECTOR_LEN];
NumericType TEMP2[MAX_VECTOR_LEN];

const static int one_i = 1;
const static int zero_i = 0;

const static NumericType one_f = 1;
const static NumericType zero_f = 0;

const char N = 'N';
const char T = 'T';

VSLStreamStatePtr stream;
#endif

void init_math_utils()
{


	std::srand(std::time(0));

#ifdef USE_MKL
	for (int i = 0;i<MAX_VECTOR_LEN;i++)
		ONES[i] = 1.0;
	vslNewStream( &stream, VSL_BRNG_MT19937, std::rand() );
#endif
}

void finish_math_utils()
{
	#ifdef USE_MKL
		/* Deleting the stream */
		vslDeleteStream( &stream );
	#endif
}

#ifdef USE_MKL
/**
 *   W  are all column major (Fortran style) matrix
 */
void vector_linear_transform(NumericType * y, int rW, int cW, const NumericType alpha, const NumericType * W, const char Trans, const NumericType * x)
{

	dgemv(&Trans, &rW,&cW, &alpha, W, &rW,x ,  &one_i, &zero_f, y, &one_i);

}


/**
 *  Y, W, X are all column major (Fortran style) matrix
 */
void matrix_dot(NumericType * Y, int rW, int cW, int rX, int cX, const NumericType alpha, const NumericType * W, const char TransW, const NumericType * X,const char TransX )
{



	if (TransW == 'N' && TransX == 'N')
	{
			assert(cW == rX);
			dgemm(&N, &N, &rW, &cX, &cW, &alpha, W, &rW, X, &cW, &zero_f, Y, &rW);
	}
	else if(TransW == 'T' && TransX == 'N')
	{
		 assert(rW == rX);
			dgemm(&T, &N, &cW, &cX, &rW, &alpha, W, &rW, X, &rW, &zero_f, Y, &cW);
	}
	else if (TransW == 'N' && TransX == 'T')
	{
		 assert(cW == cX);
		 dgemm(&N, &T, &rW, &rX, &cW, &alpha, W, &rW, X, &rX, &zero_f, Y, &rW);
	}
	else if (TransW == 'T' && TransX == 'T')
	{
		 assert(rW == cX);
		 dgemm(&T, &T, &cW, &rX, &rW, &alpha, W, &rW, X, &rX, &zero_f, Y, &cW);

	}

}

/**
 *  Y, W, X are all column major (Fortran style) matrix
 */
void matrix_linear_transform(NumericType * Y, int rW, int cW, int cX, const NumericType alpha, const NumericType * W, const char Trans, const NumericType * X, NumericType beta, const NumericType * b )
{


	char N = 'N';
	char T = 'T';
	if (Trans == 'N')
	{
			dgemm(&N, &N, &rW, &cX, &cW, &alpha, W, &rW, X, &cW, &zero_f, Y, &rW);
			dger(&rW, &cX, &beta, b, &one_i, ONES, &one_i, Y, &rW);
	}
	else
	{
			dgemm(&T, &N, &cW, &cX, &rW, &alpha, W, &rW, X, &rW, &zero_f, Y, &cW);
			dger(&cW, &cX, &beta, b, &one_i, ONES, &one_i, Y, &cW);
	}

}


void matrix_logistic_transform(NumericType * Y, int rW, int cW, int cX, const NumericType alpha, const NumericType * W, const char Trans, const NumericType * X, NumericType beta, const NumericType * b )
{
	matrix_linear_transform(Y, rW, cW,cX, alpha, W, Trans, X, beta, b );

	if (Trans == 'N')
	{
			vdExp(rW*cX,Y,Y);
			vdAdd(rW*cX,ONES,Y,Y);
			vdInv(rW*cX,Y,Y);
	}
	else
	{
			vdExp(cW*cX,Y,Y);
			vdAdd(cW*cX,ONES,Y,Y);
			vdInv(cW*cX,Y,Y);
	}

}



void fill_randn(NumericType * data, int n, NumericType mean, NumericType sigma)
{

	vdRngGaussian( VSL_METHOD_DGAUSSIAN_ICDF, stream, n, data, mean, sigma );

}

void fill_rand(NumericType * data, int n, NumericType l, NumericType h)
{
	vdRngUniform( VSL_METHOD_SUNIFORM_STD, stream, n, data, l, h );

}


#endif

