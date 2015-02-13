

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
            virtual MatrixType activate(const MatrixType & m) = 0;
            virtual MatrixType gradient(const MatrixType & m) = 0;
        };


        class linear_activator: public activator
        {
        public:

            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
        };

        class logistic_activator: public activator
        {
        public:

            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
        };

        class softmax_activator: public activator
        {
        public:

            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
        };
    }
}

#endif