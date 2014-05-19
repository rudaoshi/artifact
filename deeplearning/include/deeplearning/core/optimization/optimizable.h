#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_


#include <deeplearning/core/config.h>
#include <deeplearning/core/machine/machine.h>
#include <deeplearning/core/optimization/objective.h>

using namespace std;

namespace deeplearning
{

    template<typename InputDataSetType, typename ParameterType>
    class gradient_optimizable: public parameterized<ParameterType>，
                                public objective<InputDataSetType>
    {

    public:

      virtual ParameterType param_gradient(const InputDataSetType & testdata) = 0;

    };

}

#endif
