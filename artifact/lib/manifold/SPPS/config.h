#ifndef CONFIG_H
#define CONFIG_H

//#define USESINGLEMATH

#ifdef USESINGLEMATH

#define psFloat float
#define cublasFcopy cublasScopy
#define cublasFaxpy cublasSaxpy
#define cublasFscal cublasSscal
#define cublasFasum cublasSasum
#define cublasFgemv cublasSgemv
#define cublasFger cublasSger
#define cublasFgemm cublasSgemm
#define cublasFnrm2 cublasSnrm2
#define cublasFdot cublasSdot
#define cublasIfamin cublasIsamin
#define cublasFnrm2  cublasSnrm2

#else

#define psFloat double
#define cublasFcopy cublasDcopy
#define cublasFaxpy cublasDaxpy
#define cublasFscal cublasDscal
#define cublasFasum cublasDasum
#define cublasFgemv cublasDgemv
#define cublasFger cublasDger
#define cublasFgemm cublasDgemm
#define cublasFnrm2 cublasDnrm2
#define cublasFdot cublasDdot
#define cublasIfamin cublasIdamin
#define cublasFnrm2  cublasDnrm2
#endif

//#define _DEBUG_DETAIL

#endif