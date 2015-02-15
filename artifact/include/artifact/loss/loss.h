
#ifndef ARTIFACT_LOSSES_H
#define ARTIFACT_LOSSES_H

#include <artifact/config.h>

using namespace std;

namespace artifact
{
    namespace losses {

        class loss_function {
        public:

            virtual NumericType loss(const MatrixType &x,
                            const VectorType & y) = 0;

            /**
            * partial cost / partial x
            */
            virtual MatrixType gradient(const MatrixType &x,
                            const VectorType & y) = 0;

        };

        class mse_loss : public loss_function{
        public:

            virtual NumericType loss(const MatrixType &x,
                    const VectorType & y);

            /**
            * partial cost / partial x
            */
            virtual MatrixType gradient(const MatrixType &x,
                    const VectorType & y);

        };
    }


}

#endif
