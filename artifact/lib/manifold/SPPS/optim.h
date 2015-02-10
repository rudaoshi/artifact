#ifndef OPTIM_H_
#define OPTIM_H_

#include "config.h"
void frprmn(int n, psFloat* p, const psFloat ftol, int &iter, psFloat&fret,
	psFloat func(psFloat* ), void dfunc(psFloat*, psFloat*));

void optim_init(int maxdim);
void optim_final();

void batch_frprmn(int n, int d, psFloat* p, const psFloat ftol, int &iter, psFloat* fret,
	void func(psFloat *, psFloat* ), void dfunc(psFloat*, psFloat*));


void batch_optim_init(int maxdim,int maxn);
void batch_optim_final();

#endif