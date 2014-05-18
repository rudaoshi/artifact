#ifndef CUMATH_H_
#define CUMATH_H_

#include "spps.h"

void gausian_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* x, int M, psFloat* param, CalculationType calType);
void rowwise_gausian_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* X, int M, int N, psFloat* param, CalculationType calType);


void quadratic_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* x, int M, psFloat* param, CalculationType calType);
void rowwise_quadratic_kernel( psFloat* val, psFloat*d, psFloat* d2, psFloat* X, int M, int N, psFloat* param, CalculationType calType);



#endif