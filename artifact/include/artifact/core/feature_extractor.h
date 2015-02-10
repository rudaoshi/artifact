#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <liblearning/core/dataset.h>
#include <liblearning/core/machine.h>

namespace core
{
	class feature_extractor: public machine
	{

	public:

		virtual shared_ptr<dataset> extract_feature(const shared_ptr<dataset>  & data) = 0;

	};
}

#endif
