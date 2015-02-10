/*
 * Eigen_util.cpp
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */
#include <liblearning/util/matrix_util.h>
#include <liblearning/util/math_util.h>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <ctime>


#include <random>

#include <H5Cpp.h>
using namespace H5;
using namespace std;


template <> EigenMatrixType randn<EigenMatrixType>(int m, int n)
{

	EigenMatrixType  x(m,n);
#ifdef USE_MKL

	fill_randn(x.data(), m*n,0,1);
	
#else
	std::random_device rd;

	std::mt19937 rng(rd()); 

	std::normal_distribution<NumericType> nd(0.0, 1.0);

	std::function<double()> var_nor = std::bind(nd, rng);


    for (int i = 0;i<m;i++)
		for (int j = 0;j<n;j++)
			x(i,j) = var_nor();
	
#endif

	return x;
}





template <> EigenVectorType randn<EigenVectorType>(int size)
{
	EigenVectorType  x(size);

	std::random_device rd;

	std::mt19937 rng(rd()); 

	std::normal_distribution<NumericType> nd(0.0, 1.0);

	std::function<double()> var_nor = std::bind(nd, rng);

    for (int i = 0;i<size;i++)
			x(i) = var_nor();

	return x;
}



template <> EigenMatrixType rand<EigenMatrixType>(int m, int n)
{
	EigenMatrixType  x(m,n);

#ifdef USE_MKL

	fill_rand(x.data(), r*c,0,1);
#else

	std::random_device rd;

	std::mt19937 rng(rd()); 

	std::uniform_real_distribution<NumericType> nd(0.0, 1.0);

	std::function<double()> var_nor = std::bind(nd, rng);
	

	for (int i = 0;i<m;i++)
		for (int j = 0;j<n;j++)
			x(i,j) = var_nor();


#endif

	return x;
}

template <> EigenVectorType rand<EigenVectorType>(int size)
{
	EigenVectorType  x(size);

	std::random_device rd;

	std::mt19937 rng(rd()); 

	std::uniform_real_distribution<NumericType> nd(0.0, 1.0);

	std::function<double()> var_nor = std::bind(nd, rng);

	for (int i = 0;i<size;i++)
			x(i) = var_nor();

	return x;
}


template <> 
EigenMatrixType  operator > <EigenMatrixType > (const EigenMatrixType  & a, const EigenMatrixType  & b)
{
	assert(a.rows() == b.rows() && a.cols() == b.cols());

	EigenMatrixType  result(a.rows(),a.cols());

	for (int i = 0;i<a.rows();i++)
		for (int j = 0;j<a.cols();j++)
			result(i,j) = a(i,j) > b(i,j);

	return result;

}

#ifdef USE_GPU
template <> gpumatrix::Matrix<NumericType> randn<gpumatrix::Matrix<NumericType>>(int m, int n)
{
	return randn<EigenMatrixType>(m,n);

}

template <> gpumatrix::Vector<NumericType> randn<gpumatrix::Vector<NumericType>>(int size)
{
	return randn<EigenVectorType>(size);

}


template <> gpumatrix::Matrix<NumericType> rand<gpumatrix::Matrix<NumericType>>(int m, int n)
{
	return rand<EigenMatrixType>(m,n);

}
template <> gpumatrix::Vector<NumericType> rand<gpumatrix::Vector<NumericType>>(int size)
{
	return rand<EigenVectorType>(size);

}





template <> 
gpumatrix::Matrix<NumericType>  operator > <gpumatrix::Matrix<NumericType> > (const gpumatrix::Matrix<NumericType>  & a, const gpumatrix::Matrix<NumericType>  & b)
{
	return ((EigenMatrixType) a) > ((EigenMatrixType) b);
}

#endif

#include <liblearning/core/serialize.h>

using namespace core;

void save_matrix(const string & filename, const MatrixType & perf)
{
	H5::H5File file(filename,H5F_ACC_TRUNC);

	hsize_t data_dims[2] = { perf.rows(),perf.cols()};
	DataSpace data_space( 2, data_dims );
	H5::DataSet data = file.createDataSet("data",get_hdf_numeric_datatype<NumericType>(),data_space);

	const NumericType * data_ptr;

#ifdef USE_GPU

	EigenMatrixType temp = (EigenMatrixType)perf;
	data_ptr = temp.data();

#else
	data_ptr = perf.data();
#endif

	data.write((void *)data_ptr, get_hdf_numeric_datatype<NumericType>(), data_space, data_space );

	file.close();
}

#ifdef USE_MKL
#include <mkl_vml_functions.h>

#include <mkl_vsl.h>
#include <mkl_blas.h>
#endif

NumericType error(const MatrixType & a, const MatrixType & b)
{

#ifdef USE_MKL
	
	extern NumericType TEMP[];

	vdSub(a.size(),a.data(),b.data(),TEMP);
	int one_i = 1;
	int size = a.size();

	NumericType norm = dnrm2(&size,TEMP,&one_i);
	return norm*norm;

#else

	return (a-b).squaredNorm();

#endif
}


#ifdef USE_GPU
#include <GPUMatrix/CORE>

#endif
EigenMatrixType sqdist(const EigenMatrixType & a, const EigenMatrixType & b)
{

#if defined (USE_PARTIAL_GPU)
	gpumatrix::Matrix<NumericType> ga = a;
	gpumatrix::Matrix<NumericType> gb = b;

	gpumatrix::Matrix<NumericType> dist = -2*ga.transpose()*gb;

	gpumatrix::Vector<NumericType> aa = (ga.array()*ga.array()).colwise().sum();
	gpumatrix::Vector<NumericType> bb = (gb.array()*gb.array()).colwise().sum();

	dist.colwise() += aa;
	dist.rowwise() += bb;

	return (EigenMatrixType) dist;

//#elif defined USE_GPU
//	 VectorType aa = (a.array()*a.array()).colwise().sum();
//	 VectorType bb = (b.array()*b.array()).colwise().sum();
//	 MatrixType dist = -2*a.transpose()*b;
//
//	 dist.colwise() += aa;
//	 dist.rowwise() += bb;
//
#else
	 EigenVectorType aa = (a.array()*a.array()).colwise().sum();
	 EigenVectorType bb = (b.array()*b.array()).colwise().sum();
	 EigenMatrixType dist = -2*a.transpose()*b;

	 dist.colwise() += aa;
	 dist.rowwise() += bb.transpose();

	 //if (dist.minCoeff() < 0)
	 //{

		//MatrixType dist(a.cols(),b.cols());
		//for (int i = 0; i < dist.rows(); i ++)
		//{
		//	for (int j = 0; j < dist.cols();j++)
		//	{
		//		dist(i,j) = (a.col(i) - b.col(j)).squaredNorm();
		//	}
		//}
	 //}
	 return dist;
#endif
}

#if (defined(USE_GPU)  || defined(USE_PARTIAL_GPU) )
GPUMatrixType sqdist(const GPUMatrixType & a, const GPUMatrixType & b)
{

	gpumatrix::Matrix<NumericType> dist = -2*a.transpose()*b;

	gpumatrix::Vector<NumericType> aa = (a.array()*a.array()).colwise().sum();
	gpumatrix::Vector<NumericType> bb = (b.array()*b.array()).colwise().sum();

	dist.colwise() += aa;
	dist.rowwise() += bb;

	return  dist;
}

#endif