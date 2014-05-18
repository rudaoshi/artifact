/*
 * math_util.h
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */

#ifndef MATH_UTIL_H_
#define MATH_UTIL_H_


void init_math_utils();

void finish_math_utils();

#ifdef USE_MKL
void matrix_linear_transform(NumericType * Y, int rW, int cW, int cX, const NumericType alpha, const NumericType * W, const char Tans, const NumericType * X, NumericType beta, const NumericType * b );

void matrix_logistic_transform(NumericType * Y, int rW, int cW, int cX, const NumericType alpha, const NumericType * W, const char Tans, const NumericType * X, NumericType beta, const NumericType * b );

void matrix_dot(NumericType * Y, int rW, int cW, int rX, int cX, const NumericType alpha, const NumericType * W, const char TransW, const NumericType * X,const char TransX );

void vector_linear_transform(NumericType * y, int rW, int cW, const NumericType alpha, const NumericType * W, const char Trans, const NumericType * x);

void fill_randn(NumericType * data, int n,  NumericType mean, NumericType sigma);

void fill_rand(NumericType * data, int n, NumericType l, NumericType h);



#endif



#endif /* MATH_UTIL_H_ */
