#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <liblearning/core/dataset.h>
#include <liblearning/core/machine.h>

namespace core
{
	class classifier: public machine
	{

	public:

		virtual NumericType test(const shared_ptr<dataset>  & traindata, const shared_ptr<dataset> & data) = 0;

	};
}

#endif