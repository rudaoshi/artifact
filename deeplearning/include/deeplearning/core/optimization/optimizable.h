#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_


#include <deeplearning/core/config.h>

#include <tuple>
using namespace std;

namespace deeplearning
{

template<typename InputDataSetType, typename ParameterType>
class gradient_optimizable: public parameterized<ParameterType>
{
  
public:

  virtual NumicalType objective(const InputDataSetType & traindata) = 0;

  virtual ParameterType param_gradient(const InputDataSetType & testdata) = 0;

};

}

#endif
