

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

            virtual RowVectorType activate(const RowVectorType & v) = 0;
            virtual RowVectorType gradient(const RowVectorType & v) = 0;
            virtual RowVectorType gradient(const RowVectorType & v,
                    const RowVectorType & activated) = 0;
            virtual MatrixType activate(const MatrixType & m) = 0;
            virtual MatrixType gradient(const MatrixType & m) = 0;
            virtual MatrixType gradient(const MatrixType & v,
                    const MatrixType & activated) = 0;
        };


        class linear_activator: public activator
        {
        public:

            virtual RowVectorType activate(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v,
                    const RowVectorType & activated);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m,
                    const MatrixType & activated);
        };

        class logistic_activator: public activator
        {
        public:

            virtual RowVectorType activate(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v,
                    const RowVectorType & activated);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m,
                    const MatrixType & activated);
        };

        class relu_activator: public activator
        {
        public:

            virtual RowVectorType activate(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v,
                                           const RowVectorType & activated);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m,
                                        const MatrixType & activated);
        };

        class softmax_activator: public activator
        {
        public:

            virtual RowVectorType activate(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v);
            virtual RowVectorType gradient(const RowVectorType & v,
                    const RowVectorType & activated);
            virtual MatrixType activate(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m);
            virtual MatrixType gradient(const MatrixType & m,
                    const MatrixType & activated);
        };
    }
}

#endif