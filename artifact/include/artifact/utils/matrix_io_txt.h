#ifndef ARTIFACT_UTILS_MATRIX_IO_TXT_H
#define ARTIFACT_UTILS_MATRIX_IO_TXT_H

#include <string>
using namespace std;

#include <artifact/config.h>

namespace artifact
{
    namespace utils
    {
        MatrixType load_matrix_from_txt(const string & file_name);
        VectorType load_vector_from_txt(const string & file_name);
    }
}


#endif