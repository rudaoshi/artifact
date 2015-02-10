
#ifndef SPPS_H_
#define SPPS_H_

#include "config.h"
//typedef struct _TempVars
//{
//
//	psFloat* x;
//	psFloat* diffVecs;
//	psFloat* normDiffV;
//	psFloat* kernelVal;
//	psFloat* kernelD;
//	psFloat* kernelD2;
//	
//	psFloat* b;
//	psFloat* Jbx;
//	psFloat* Jbs;
//	psFloat* p;
//	psFloat* Jpx;
//	psFloat* Jps;
//	
//	psFloat* q;
//	
//	psFloat* Hbxx;
//	psFloat* Hbxs;
//	
//	psFloat* mx;
//	psFloat* Jmx;
//	psFloat* Jms;
//	
//	psFloat* y;
//	psFloat dis;
//	psFloat* Jdx;
//	psFloat* Hdxx;
//	psFloat* Jds;
//	psFloat* Hdxs;
//
//	psFloat* gx;     // 求投影坐标时，对x进行共轭梯度优化时使用的g
//	psFloat* hx;     // 求投影坐标时，对x进行共轭梯度优化时使用的h
//	
//	
//
//		
//} TempVars;

typedef enum _KernelType
{
	Gaussian, Quadratic
} KernelType ;



typedef struct _SPPS
{
	psFloat* Y; // the training data
	int N;     // the number of the data 
	int M;     // the number of the baises;
	int D;     // the observable dimensionality
    int d;     // the feature dimensionality
    psFloat* X;   // the feature of the data
    psFloat* reconY;  // the reconstruction points;
    
    psFloat* T ;   // Control points in target space
    psFloat* S ;   // Control points in source space
    psFloat* Sigma ; // Kernel width
    
    KernelType kernel ;  // kernel types;

	int * nngraph;   // the neighborhood graph, N*MAX_NEIGHBOR matrix

	int neighbor_pair;
    
//    TempVars tempVars;

} SPPS;

typedef enum _CalculationType
{
	Eval, Jacobbi, Hessian
} CalculationType;


void SPPS_init(psFloat* Y, int D, int N, psFloat* initS, psFloat* initT, int d, int M, KernelType kernel, psFloat* Sigma, int * nngraph, int max_neighbor);
void SPPS_final();

void slps_map_train(psFloat * _mX);


void slps_project_train(psFloat * _X );



#endif 
