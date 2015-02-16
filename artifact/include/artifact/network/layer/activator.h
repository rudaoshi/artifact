

#ifndef ARTIFACT_NETWORK_LAYER_ACTIVATOR_H
#define ARTIFACT_NETWORK_LAYER_ACTIVATOR_H

#include <string>
using namespace std;

#include <artifact/config.h>

namespace artifact
{
    namespace network
    {
        class activator
        {
        public:

            virtual VectorType activate(const VectorType & v) = 0;
            virtual VectorType gradient(const VectorType & v) = 0;
            virtual VectorType gradient(const VectorType & v,
                    const VectorType & activated) = 0;
            virtual MatrixType activate(const MatrixType & m) = 0;
            virtual MatrixType gradient(const MatrixType & m) = 0;
            virtual MatrixType gradient(const MatrixType & v,
                    const MatrixType & activated) = 0;
        };


        class linear_activator: public activator
        {
        public:

            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual VectorType gradient(const VectorType & v,
                    const VectorType & activated);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m,
                    const MatrixType & activated);
        };

        class logistic_activator: public activator
        {
        public:

            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual VectorType gradient(const VectorType & v,
                    const VectorType & activated);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m,
                    const MatrixType & activated);
        };

        class softmax_activator: public activator
        {
        public:

            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual VectorType gradient(const VectorType & v,
                    const VectorType & activated);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m,
                    const MatrixType & activated);
        };
    }
}

#endif