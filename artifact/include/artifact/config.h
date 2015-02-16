#ifndef ARTIFACT_CONFIG_H
#define ARTIFACT_CONFIG_H

//#include <boost/serialization/array.hpp>
//#include <boost/serialization/collection_size_type.hpp>
//#include <boost/serialization/nvp.hpp>


//#define EIGEN_MATRIX_PLUGIN <artifact/util/Eigen_matrix_addon.h>

#define BOOST_FILESYSTEM_VERSION 3

#undef USE_MKL

#undef USE_GPU

#undef USE_PARTIAL_GPU

#undef USE_MATLAB

#ifdef USE_GPU

    #include <gpumatrix/CORE>
    using namespace gpumatrix;
    #define NumericType double
    #define MatrixType Matrix<NumericType>
    #define VectorType Vector<NumericType>


    #define InterfaceMatrixType MatrixType
    #define InterfaceVectorType VectorType

    #define EigenMatrixType Eigen::MatrixXd
    #define EigenVectorType Eigen::VectorXd

    #define MatrixType gpumatrix::Matrix<NumericType>
    #define VectorType gpumatrix::Vector<NumericType>

    #define GPUMatrixType gpumatrix::Matrix<NumericType>
    #define GPUVectorType gpumatrix::Vector<NumericType>

    using gpumatrix::Map;

#elif defined USE_PARTIAL_GPU

    #include <Eigen/Core>
    #include <Eigen/QR>
    #include <gpumatrix/CORE>

    #define NumericType double

    #define MatrixType Eigen::MatrixXd
    #define VectorType Eigen::VectorXd

    #define InterfaceMatrixType MatrixType
    #define InterfaceVectorType VectorType

    #define EigenMatrixType Eigen::MatrixXd
    #define EigenVectorType Eigen::VectorXd

    #define GPUMatrixType gpumatrix::Matrix<NumericType>
    #define GPUVectorType gpumatrix::Vector<NumericType>

    using Eigen::Map;

#else
	#include <Eigen/Core>
    #include <Eigen/QR>
	using namespace Eigen;
	#define NumericType double
	#define MatrixType MatrixXd
	#define VectorType VectorXd

	#define EigenMatrixType Eigen::MatrixXd
	#define EigenVectorType Eigen::VectorXd
	#define InterfaceMatrixType MatrixXd
	#define InterfaceVectorType VectorXd

	using Eigen::Map;

#endif


#endif
