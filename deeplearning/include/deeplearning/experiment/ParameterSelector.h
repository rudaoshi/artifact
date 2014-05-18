#ifndef PARAMETER_SELECTOR_H
#define PARAMETER_SELECTOR_H

#include <boost/singals2.hpp>

namespace experiment
{
	class ParameterSelector
	{
	  shared_ptr<data_set> train;
	  shared_ptr<data_set> valid;

	  param_generator generator;

	public:
	  boost::signals2::signal<> machineBuild;


	};
}

#endif
