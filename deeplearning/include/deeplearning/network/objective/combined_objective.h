/*
* combined_objective.h
*
*  Created on: 2010-6-19
*      Author: sun
*/

#ifndef COMBINED_OBJECTIVE_H_
#define COMBINED_OBJECTIVE_H_

#include "data_related_network_objective.h"

#include <vector>
namespace deep
{
	namespace objective
	{
		class POCO_EXPORT  combined_objective: public data_related_network_objective
		{
			std::vector<shared_ptr<network_objective>> objs;

			std::vector<NumericType> weights;

		protected:

			virtual NumericType prepared_value(deep_auto_encoder & net) ;
			virtual vector<shared_ptr<MatrixType>> prepared_value_delta(deep_auto_encoder & net) ;


		public:
			combined_objective();
			combined_objective(const combined_objective & obj);
			virtual ~combined_objective();

			void add_objective( const shared_ptr<network_objective> & obj, NumericType weight);

			void set_weights(const std::vector<NumericType> & weights);

			void set_weight(NumericType weight, int index);

			virtual void set_dataset(const shared_ptr<dataset> & data_set);


			virtual tuple<NumericType, VectorType> value_diff(deep_auto_encoder & net) ;

			virtual NumericType value(deep_auto_encoder & net)  ;

			virtual combined_objective * clone();

			virtual string get_info();

		};
	}
}
#endif /* COMBINED_OBJECTIVE_H_ */
