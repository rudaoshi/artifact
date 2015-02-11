#ifndef FISHER_DECODER_OBJECTIVE_H_
#define FISHER_DECODER_OBJECTIVE_H_


#include "data_related_network_objective.h"
namespace deep
{
	namespace objective
	{
class POCO_EXPORT  fisher_objective: public data_related_network_objective
{
	MatrixType Aw;
	MatrixType Ab;

	NumericType trSw;
	NumericType trSb;

	MatrixType Aw_diff_helper;
	MatrixType Ab_diff_helper;

	double current_obj_val;

public:
	fisher_objective();
	virtual ~fisher_objective();

	virtual void set_dataset(const shared_ptr<dataset> & data_set);

	virtual NumericType prepared_value(deep_auto_encoder & net) ;
	virtual vector<shared_ptr<MatrixType>> prepared_value_delta(deep_auto_encoder & net) ;

	virtual fisher_objective * clone();

	virtual string get_info();

};
	}
}
#endif /* MSE_DECODER_OBJECTIVE_H_ */

