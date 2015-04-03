/*
 * conjugate_gradient_optimizer.h
 *
 *  Created on: 2010-6-20
 *      Author: sun
 */

#ifndef ARTIFACT_OPTIMIZATION_CGD_OPTIMIZER_H_
#define ARTIFACT_OPTIMIZATION_CGD_OPTIMIZER_H_
#include <artifact/config.h>
#include <artifact/optimization/optimizer.h>

namespace artifact {
    namespace optimization {


        class cgd_optimizer : public optimizer {
        public:
            NumericType ftol;
            int max_epoches;

        public:
            cgd_optimizer();

            virtual ~cgd_optimizer(void);

            virtual VectorType optimize(
                    optimizable &obj,
                    const VectorType &param0,
                    const MatrixType &X,
                    const MatrixType *y = nullptr  // nullptr for unsupervised optimizer
            );

        };
    }
}

#endif /* ARTIFACT_OPTIMIZATION_CGD_OPTIMIZER_H_ */
