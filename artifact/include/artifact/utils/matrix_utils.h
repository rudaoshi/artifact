/*
 * blitz_util.h
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */

#ifndef ARTIFACT_UTIL_MATRIX_UTIL_H_
#define ARTIFACT_UTIL_MATRIX_UTIL_H_



#include <artifact/config.h>


namespace artifact
{
    namespace utils
    {

        template <class MT> MT randn(int r, int c);

        template <class VT> VT randn(int size);

        template <class MT> MT rand(int r, int c);

        template <class VT> VT rand(int size);

//MatrixType randn(int r, int c);
//
//VectorType randn(int size);
//
//MatrixType rand(int r, int c);
//
//VectorType rand(int size);

        template <class MT>
        MT operator > (const MT & a, const MT & b);

        NumericType error(const MatrixType & a, const MatrixType & b);

        EigenMatrixType sqdist(const EigenMatrixType & a, const EigenMatrixType & b);

        #if (defined(USE_GPU)  || defined(USE_PARTIAL_GPU) )
        GPUMatrixType sqdist(const GPUMatrixType & a, const GPUMatrixType & b);
        #endif


        template <typename M> M logistic_func(const M & m)
        {
        #ifdef USE_GPU

            return m.array().logistic();

        #elif defined USE_PARTIAL_GPU

            return m.array().logistic();

        #else

                    return (1 + (-m).array().exp()).inverse();

        #endif
        }

        EigenMatrixType load_matrix_from_txt(const string & file_name);
        EigenVectorType load_vector_from_txt(const string & file_name);


    }
}
#endif /* BLITZ_UTIL_H_ */