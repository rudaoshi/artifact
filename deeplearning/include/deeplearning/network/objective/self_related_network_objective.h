

#ifndef SELF_RELATED_NETWORK_OBJECTIVE_H
#define SELF_RELATED_NETWORK_OBJECTIVE_H


#include "../network_objective.h"

namespace deep
{
	namespace objective
	{
		class self_related_network_objective:public network_objective
		{
		protected:

		public:
			self_related_network_objective();
			virtual ~self_related_network_objective();

			virtual void set_dataset(const shared_ptr<dataset> & data_set);
			virtual tuple<NumericType, VectorType> value_diff(deep_auto_encoder & net)  = 0;

			virtual NumericType value(deep_auto_encoder & net)  = 0;


			virtual self_related_network_objective * clone() = 0;

		};
	}
}
#endif /* OBJECTIVE_H_ */


