/*
 * objective.h
 *
 *  Created on: 2010-6-17
 *      Author: sun
 */

#ifndef OBJECTIVE_H_
#define OBJECTIVE_H_


#include <liblearning/core/dataset.h>

#include <string>
#include <tuple>
using namespace std;

#include <liblearning/deep/network_objective_type.h>

namespace deep
{
	class deep_auto_encoder;
	using namespace core;
	class network_objective
	{
	protected:
		network_objective_type type;

	public:
		network_objective();
		virtual ~network_objective();

		virtual void set_dataset(const shared_ptr<dataset> & data_set) = 0;

		virtual network_objective_type get_type()const {return type;};

		virtual tuple<NumericType, VectorType> value_diff(deep_auto_encoder & net)  = 0 ;

		virtual NumericType value(deep_auto_encoder & net)  = 0 ;

		virtual network_objective * clone() = 0;

		virtual string get_info() = 0;
	};
}

#endif /* OBJECTIVE_H_ */
