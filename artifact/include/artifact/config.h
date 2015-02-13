#ifndef CONFIG_H
#define CONFIG_H

#include <boost/serialization/array.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/nvp.hpp>


#define EIGEN_MATRIX_PLUGIN <artifact/util/Eigen_matrix_addon.h>

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

	#define MatrixType MatrixXd
	#define VectorType VectorXd

	using Eigen::Map;

#endif


//	//#include <Eigen/Core>
//	//using namespace Eigen;
//	//#define NumericType double
//	//#define MatrixType MatrixXd
//	//#define VectorType VectorXd
//
//#ifdef USE_GPU
//
//	#include <gpumatrix/CORE>
//
//#define InterfaceMatrixType MatrixType
//#define InterfaceVectorType VectorType
//
//#define EigenMatrixType Eigen::MatrixXd
//#define EigenVectorType Eigen::VectorXd
//
//#define MatrixType gpumatrix::Matrix<NumericType>
//#define VectorType gpumatrix::Vector<NumericType>
//
//using gpumatrix::Map;
//#else
//#define InterfaceMatrixType MatrixXd
//#define InterfaceVectorType VectorXd
//
//#define MatrixType MatrixXd
//#define VectorType VectorXd
//#endif
//
//#include <pocodefs.h>

#if  defined(_MSC_VER) && (_MSC_VER == 1500)

#include <memory>
#include <tuple>
#include <functional>
using std::tr1::shared_ptr;
using std::tr1::dynamic_pointer_cast;
using std::tr1::tuple;
using std::tr1::make_tuple;
using std::tr1::tie;
using std::tr1::function;
#define nullptr 0
#include <math.h>
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define fpu_error(x) (isinf(x) || isnan(x))

#elif (defined(_MSC_VER) && (_MSC_VER == 1600))

#include <memory>
#include <tuple>
#include <functional>
using std::shared_ptr;
using std::dynamic_pointer_cast;
using std::tuple;
using std::make_tuple;
using std::tie;
using std::function;
#elif defined (__GNUC__)
#include <memory>
#include <tuple>
#include <functional>
using std::shared_ptr;
using std::dynamic_pointer_cast;
using std::tuple;
using std::make_tuple;
using std::tie;
using std::function;
#else
	#error("Unknown Compiler");
#endif

#define POCO_EXPORT

//template <typename T> CUDPPDatatype get_cudpp_datatype();
//
//template<> CUDPPDatatype get_cudpp_datatype<double>()
//{
//	return CUDPP_DOUBLE;
//}
//
//template<> CUDPPDatatype get_cudpp_datatype<float>()
//{
//	return CUDPP_FLOAT;
//}

#endif
