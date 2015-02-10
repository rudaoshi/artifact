#pragma once
#include "dataset_transform.h"

namespace transform
{

	class POCO_EXPORT unit_interval_transform :
		public dataset_transform
	{
		NumericType min_elem;
		NumericType max_elem;


	public:
		unit_interval_transform(const dataset & train);
		unit_interval_transform(NumericType min_elem_, NumericType max_elem);

		virtual ~unit_interval_transform(void);

		virtual shared_ptr<dataset> apply(const dataset & data);
	};

}

