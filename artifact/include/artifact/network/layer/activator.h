

#ifndef ARTIFACT_NETWORK_LAYER_ACTIVATOR_H
#define ARTIFACT_NETWORK_LAYER_ACTIVATOR_H


#include <artifact/config.h>

namespace artifact
{
    namespace netowrk
    {
        class activator
        {
        public:
            const static string type = "base";

            virtual VectorType activate(const VectorType & v) = 0;
            virtual VectorType gradient(const VectorType & v) = 0;
            virtual MatrixType activate(const MatrixType & m) = 0;
            virtual MatrixType gradient(const MatrixType & m) = 0;
        };


        class linear_activator: public activator
        {
        public:
            const static string type = "linear";
            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
        };

        class logistic_activator: public activator
        {
        public:
            const static string type = "logistic";
            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
        };

        class softmax_activator: public activator
        {
        public:
            const static string type = "softmax";
            virtual VectorType activate(const VectorType & v);
            virtual VectorType gradient(const VectorType & v);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
        };
    }
}

#endif