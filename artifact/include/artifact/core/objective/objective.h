
#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_

#include <artifact/config.h>

#include <tuple>
using namespace std;

namespace artifact
{
    namespace core {

        class objective {
        public:

            virtual NumericType cost(const MatrixType &x) = 0;

            virtual MatrixType cost_gradient(const MatrixType &x) = 0;

        };
    }


}

#endif
