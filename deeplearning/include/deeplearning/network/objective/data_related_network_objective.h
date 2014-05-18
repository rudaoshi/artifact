/*
* objective.h
*
*  Created on: 2010-6-17
*      Author: sun
*/

#ifndef DATA_RELATED_NETWORK_OBJECTIVE_H
#define DATA_RELATED_NETWORK_OBJECTIVE_H


#include <liblearning/core/dataset.h>


#include <tuple>
using namespace std;

#include "../network_objective.h"


namespace deep
{
	class deep_auto_encoder;
	namespace objective
	{
		using namespace core;
		class data_related_network_objective:public network_objective
		{
		protected:

			shared_ptr<dataset> data_set;

		public:

		public:
			data_related_network_objective();
			virtual ~data_related_network_objective();

			virtual void set_dataset(const shared_ptr<dataset> & data_set_);

			virtual NumericType prepared_value(deep_auto_encoder & net)  = 0 ;
			virtual vector<shared_ptr<MatrixType>> prepared_value_delta(deep_auto_encoder & net)  = 0 ;

			virtual tuple<NumericType, VectorType> value_diff(deep_auto_encoder & net) ;

			virtual NumericType value(deep_auto_encoder & net) ;


			virtual data_related_network_objective * clone() = 0;


		};
	}
}
#endif /* OBJECTIVE_H_ */
