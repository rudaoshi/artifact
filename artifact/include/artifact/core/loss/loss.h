
#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_

#include <artifact/config.h>

#include <tuple>
using namespace std;

namespace artifact
{
    namespace core {

        class loss {
        public:

            virtual NumericType loss(const MatrixType &x,
                            const VectorType & y) = 0;

            /**
            * partial cost / partial x
            */
            virtual MatrixType gradient(const MatrixType &x,
                            const VectorType & y) = 0;

        };
    }


}

#endif
