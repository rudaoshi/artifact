#ifndef ARTIFACT_UTILS_MATRIX_UTILS_H
#define ARTIFACT_UTILS_MATRIX_UTILS_H

namespace artifact
{
    namespace utils
    {
        tuple<MatrixType, VectorType> shuffle_data_set(const MatrixType & X,
                const VectorType & y);
    }
}


#endif