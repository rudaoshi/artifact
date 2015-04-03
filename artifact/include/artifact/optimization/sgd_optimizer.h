#ifndef ARTIFACT_OPTIMIZATION_SGD_OPTIMIZATION_H
#define ARTIFACT_OPTIMIZATION_SGD_OPTIMIZATION_H



#include <artifact/optimization/optimizer.h>


namespace artifact
{
    namespace optimization
    {


        class sgd_optimizer :
                public optimizer
        {
        public:
            NumericType learning_rate;
            NumericType decay_rate;
            int batch_size;
            int max_epoches;

        public:
            sgd_optimizer();
            virtual ~sgd_optimizer(void);

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
