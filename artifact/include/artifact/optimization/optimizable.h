#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_


#include <algorithm>
using namespace std;


#include <artifact/core/config.h>
#include <artifact/core/machine/machine.h>
#include <artifact/core/objective/objective.h>



namespace artifact
{

    namespace core
    {


        class optimizable
        {
        public:

            virtual VectorType  get_parameter() = 0;
            virtual void set_parameter(const VectorType &parameter_) = 0;

            virtual NumericType objective(const MatrixType & x,
                    const VectorType & y) = 0;

        };

        /**
        * gradient_optimizable:
        * an interface that alow gradient base optimization method can be applied
        */
        class gradient_optimizable: public optimizable
        {

        public:
            /**
            * partial output/partial param
            */
            virtual pair<NumericType, VectorType> gradient(const MatrixType & x,
                    const VectorType & y) = 0;

        };
    }
}

#endif
