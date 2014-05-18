/*
* mse_decoder_objective.h
*
*  Created on: 2010-6-18
*      Author: sun
*/

#ifndef MSE_DECODER_OBJECTIVE_H_
#define MSE_DECODER_OBJECTIVE_H_


#include "data_related_network_objective.h"

namespace deep
{
	namespace objective
	{
		class POCO_EXPORT mse_objective:public data_related_network_objective
		{

			double cur_obj_val;

		public:
			mse_objective();
			virtual ~mse_objective();

			virtual NumericType prepared_value(deep_auto_encoder & net) ;
			virtual vector<shared_ptr<MatrixType>> prepared_value_delta(deep_auto_encoder & net) ;

			virtual mse_objective * clone();

			virtual string get_info();
		};
	}
}
#endif /* MSE_DECODER_OBJECTIVE_H_ */
