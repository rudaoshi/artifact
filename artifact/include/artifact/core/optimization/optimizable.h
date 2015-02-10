#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_


#include <artifact/core/config.h>
#include <artifact/core/machine/machine.h>
#include <artifact/core/objective/objective.h>

using namespace std;

namespace artifact
{

    namespace core
    {


        /**
        * gradient_optimizable:
        * an interface that alow gradient base optimization method can be applied
        */
        class gradient_optimizable
        {

        public:

            virtual const VectorType &get_parameter() = 0;
            virtual void set_parameter(const VectorType &parameter_) = 0;

            virtual NumericType objective(const MatrixType & x,
                    const VectorType & y) = 0;
            /**
            * partial output/partial param
            */
            virtual VectorType gradient(const MatrixType & x,
                    const VectorType & y) = 0;

        };
    }
}

#endif
