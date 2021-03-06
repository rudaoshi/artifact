#ifndef ARTIFACT_OPTIMIZATION_GD_OPTIMIZATION_H
#define ARTIFACT_OPTIMIZATION_GD_OPTIMIZATION_H



#include <artifact/optimization/optimizer.h>


namespace artifact
{
    namespace optimization
    {

        class gd_optimizer :
                public optimizer
        {
        public:
            NumericType learning_rate;
            NumericType decay_rate;
            int max_epoches;

        public:
            gd_optimizer();
            virtual ~gd_optimizer(void);

            virtual VectorType optimize(
                    optimizable & obj,
                    const VectorType & param0,
                    const MatrixType & X,
                    const MatrixType * y = nullptr  // nullptr for unsupervised optimizer
                    );

        };

    }

}

#endif
