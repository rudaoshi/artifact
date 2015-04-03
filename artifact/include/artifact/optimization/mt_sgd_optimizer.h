#ifndef ARTIFACT_OPTIMIZATION_MT_SGD_OPTIMIZATION_H
#define ARTIFACT_OPTIMIZATION_MT_SGD_OPTIMIZATION_H



#include <artifact/optimization/optimizer.h>


namespace artifact
{
    namespace optimization
    {


        class mt_sgd_optimizer :
                public optimizer
        {
        public:
            NumericType learning_rate;
            NumericType decay_rate;
            int thread_num;
            int batch_per_thread;
            int max_epoches;

        public:
            mt_sgd_optimizer();
            virtual ~mt_sgd_optimizer(void);

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
