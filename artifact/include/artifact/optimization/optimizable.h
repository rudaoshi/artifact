#ifndef ARTIFACT_OPTIMIZABLE_H_
#define ARTIFACT_OPTIMIZABLE_H_


#include <tuple>
using namespace std;


#include <artifact/config.h>


namespace artifact
{

    namespace optimization
    {


        class optimizable
        {
        public:

            virtual VectorType  get_parameter() = 0;
            virtual void set_parameter(const VectorType &parameter_) = 0;

            virtual NumericType objective(const MatrixType & x,
                    const MatrixType * y = 0) = 0;

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
            virtual tuple<NumericType, VectorType> gradient(const MatrixType & x,
                    const MatrixType * y = 0) = 0;

        };
    }
}

#endif
