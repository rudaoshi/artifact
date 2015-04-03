/*
 * optimizer.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef ARTIFACT_OPTIMIZER_H_
#define ARTIFACT_OPTIMIZER_H_


#include <tuple>

using namespace std;

#include <artifact/optimization/optimizable.h>

namespace artifact
{
    namespace optimization
    {

        class optimizer
        {
        public:

            virtual VectorType optimize(
                    optimizable & obj,
                    const VectorType & param0,
                    const MatrixType & X,
                    const MatrixType * y = nullptr  // nullptr for unsupervised optimizer
                    ) = 0;

        };


    }
}

#endif /* OPTIMIZER_H_ */
