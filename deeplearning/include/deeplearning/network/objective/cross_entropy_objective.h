/*
* mse_decoder_objective.h
*
*  Created on: 2010-6-18
*      Author: sun
*/

#ifndef CROSS_ENTROPY_OBJECTIVE_H_
#define CROSS_ENTROPY_OBJECTIVE_H_


#include "data_related_network_objective.h"

namespace deep
{
	namespace objective
	{
		class POCO_EXPORT cross_entropy_objective:public data_related_network_objective
		{

			double current_objective_val;

		public:
			cross_entropy_objective();
			virtual ~cross_entropy_objective();

			virtual NumericType prepared_value(deep_auto_encoder & net) ;
			virtual vector<shared_ptr<MatrixType>> prepared_value_delta(deep_auto_encoder & net) ;

			virtual cross_entropy_objective * clone();

			virtual string get_info();
		};
	}
}
#endif /* MSE_DECODER_OBJECTIVE_H_ */
