
#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_

#include <deeplearning/core/config.h>

#include <tuple>
using namespace std;

namespace deeplearning
{
  template<typename InputDataSetType>
	class objective
	{
	public:

		virtual NumericType cost(const InputDataSetType & x) = 0;
		virtual InputDataSetType cost_gradient(const InputDataSetType & x) = 0;

	};


}

#endif
