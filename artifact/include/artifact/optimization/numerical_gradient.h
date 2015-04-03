#ifndef ARTIFACT_NUMERICAL_GRADIENT_H_
#define ARTIFACT_NUMERICAL_GRADIENT_H_


#include <algorithm>
using namespace std;


#include <artifact/config.h>
#include <artifact/optimization/optimizable.h>

namespace artifact {

    namespace optimization {

        VectorType numerical_gradient(optimizable & machine, const VectorType & param, const MatrixType & X, const MatrixType * y);

    }
}

#endif