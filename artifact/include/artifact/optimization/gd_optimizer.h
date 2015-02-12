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
            NumericType learning_rate;
            NumericType decay_rate;

            int max_epoches;

        public:
            gd_optimizer(NumericType learning_rate, NumericType decay_rate, int max_epoches);
            virtual ~gd_optimizer(void);

            virtual tuple<NumericType, VectorType> optimize(
                    optimizable & obj,
                    const VectorType & param0,
                    const MatrixType & X,
                    const VectorType * y = nullptr  // nullptr for unsupervised optimizer
                    );

        };

    }

}

#endif
