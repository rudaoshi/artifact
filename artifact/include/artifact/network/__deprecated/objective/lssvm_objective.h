#ifndef LSSVM_OBJECTIVE_H_
#define LSSVM_OBJECTIVE_H_

#include <liblearning/classifier/LeastSquareSVM.h>

#include "data_related_network_objective.h"
namespace deep
{
	namespace objective
	{
		
		using namespace classification;
		
class POCO_EXPORT  lssvm_objective:public data_related_network_objective
{

	LeastSquareSVM svm;
	
	double current_obj_val;

public:
	lssvm_objective(NumericType gammar);
	virtual ~lssvm_objective();

	virtual void set_dataset(const shared_ptr<dataset> & data_set);

	virtual NumericType prepared_value(deep_auto_encoder & net) ;
	virtual vector<shared_ptr<MatrixType>> prepared_value_delta(deep_auto_encoder & net) ;

	virtual lssvm_objective * clone();

	virtual string get_info();

};
	}
}
#endif /* LSSVM_OBJECTIVE_H_ */

