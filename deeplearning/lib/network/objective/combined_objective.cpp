/*
 * combined_objective.cpp
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#include <liblearning/deep/objective/combined_objective.h>

#include <liblearning/deep/deep_auto_encoder.h>

#include <liblearning/deep/objective/self_related_network_objective.h>

using namespace deep;
using namespace deep::objective;

combined_objective::combined_objective()
{
	type = self_related;
}

combined_objective::~combined_objective()
{
}


combined_objective::combined_objective(const combined_objective & obj):data_related_network_objective(obj),objs(obj.objs.size()),weights(obj.weights)
{
	for (int i = 0;i<obj.objs.size();i++)
	{
		objs[i].reset(obj.objs[i]->clone());
	}
}

void combined_objective::add_objective(const shared_ptr<network_objective> & obj,NumericType weight)
{
	if (type < obj->get_type())
		type = obj->get_type();

	objs.push_back(obj);
	weights.push_back(weight);
}

void combined_objective::set_weights(const std::vector<NumericType> & weights_)
{
	weights = weights_;
}

void combined_objective::set_weight(NumericType weight, int index)
{
	weights[index] = weight;
}

void combined_objective::set_dataset(const shared_ptr<dataset> & data_set_)
{
	data_set = data_set_;
	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<data_related_network_objective > p_obj =  dynamic_pointer_cast<data_related_network_objective> (objs[i]);
		if (p_obj)
			p_obj->set_dataset(data_set_);
	}
}

NumericType combined_objective::value(deep_auto_encoder & net)
{
	NumericType value =  data_related_network_objective::value(net);


	// add the objective of self related objectives
	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<self_related_network_objective> p_obj =  dynamic_pointer_cast<self_related_network_objective > (objs[i]);
		if (p_obj)
			value += weights[i]*objs[i]->value(net);
	}

	return value;

}

tuple<NumericType, VectorType> combined_objective::value_diff(deep_auto_encoder & net)
{
	NumericType value = 0;
	VectorType value_diff;



	tie(value,value_diff) = data_related_network_objective::value_diff(net);

	// add the objective of self related objectives
	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<self_related_network_objective> p_obj =  dynamic_pointer_cast<self_related_network_objective > (objs[i]);
		if (p_obj )
		{
			NumericType cur_value = 0;
			VectorType cur_value_diff;

			tie(cur_value,cur_value_diff) = objs[i]->value_diff(net);

			value +=  weights[i]*cur_value;
			value_diff +=  weights[i] * cur_value_diff;
		}
	}

	return make_tuple(value,value_diff);

}




NumericType combined_objective::prepared_value(deep_auto_encoder & net)
{

	NumericType value = 0;

	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<data_related_network_objective > p_obj =  dynamic_pointer_cast<data_related_network_objective> (objs[i]);
		if (p_obj)
		{
//			std::cout<< "Adding " << i << "-th object value" << std::endl;
			value += weights[i]*p_obj->prepared_value(net);
//			std::cout<< "Finish Adding " << i << "-th object value" << std::endl;
		}
	}

	if (value < 0)
	{
		std::cout << "Error Occured!" << std::endl;
		net.save_hdf("bad_machine.hdf");

		std::getchar();
	}
	return value;
}

vector<shared_ptr<MatrixType>> combined_objective::prepared_value_delta(deep_auto_encoder & net)
{

	NumericType value = 0;
	vector<shared_ptr<MatrixType>> value_diff(2);

	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<data_related_network_objective > p_obj =  dynamic_pointer_cast<data_related_network_objective> (objs[i]);
		if (p_obj )
		{
//			std::cout<< "Computing  " << i << "-th diff value" << std::endl;
			vector<shared_ptr<MatrixType>> cur_diff = p_obj->prepared_value_delta(net);

//			std::cout<< "Adding  " << i << "-th diff value" << std::endl;

			for ( int j = 0;j < 2;j++)
			{
				if (cur_diff[j])
				{
					if (value_diff[j])
						(*value_diff[j]) +=  weights[i]*(* cur_diff[j]);
					else
						value_diff[j].swap(cur_diff[j]);
				}
			}

//			std::cout<< "Finish Adding " << i << "-th object diff value" << std::endl;
		}

	}

	return value_diff;
}

#include <iostream>


combined_objective * combined_objective::clone()
{
	return new combined_objective(*this);
}

string combined_objective::get_info()
{
	string info;
	for (int i = 0;i<objs.size();i++)
	{
		info += objs[i]->get_info();
		info += "\n";
	}
	return info;
}
