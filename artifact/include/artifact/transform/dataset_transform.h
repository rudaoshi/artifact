#ifndef DATASET_PREPROCESSOR_H
#define DATASET_PREPROCESSOR_H

#include <memory>

using namespace std;

#include <liblearning/core/dataset.h>

namespace transform
{
	using namespace core;
	class dataset_transform
	{
	public:
		dataset_transform(void);
		virtual ~dataset_transform(void);

		virtual shared_ptr<dataset> apply(const dataset & data) = 0;
	};
}

#endif

