
#ifndef RIDGE_REGRESSION_REGULARIZOR_H
#define RIDGE_REGRESSION_REGULARIZOR_H

#include "self_related_network_objective.h"

namespace deep
{
	namespace objective
	{
		class POCO_EXPORT ridge_regression_regularizor :
			public self_related_network_objective
		{
			double cur_obj_val;
		public:
			ridge_regression_regularizor(void);
			virtual ~ridge_regression_regularizor(void);


			virtual tuple<NumericType, VectorType> value_diff(deep_auto_encoder & net);

			virtual NumericType value(deep_auto_encoder & net);

			virtual ridge_regression_regularizor * clone();

			virtual string get_info();
		};
	}
}
#endif