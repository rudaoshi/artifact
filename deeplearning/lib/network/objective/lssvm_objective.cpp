
#include <liblearning/deep/objective/lssvm_objective.h>

#include <liblearning/core/supervised_dataset.h>
#include <liblearning/deep/deep_auto_encoder.h>
#include <liblearning/kernel/linear_kernel.h>


using namespace deep;
using namespace deep::objective;
using namespace kernelmethod;

lssvm_objective::lssvm_objective(NumericType gammar)
{
	type = encoder_related;
	
    LeastSquareSVMParam param;
	
    param.gamma = gammar;
	
    param.kernelfunc.reset ( new linear_kernel );
	
	svm.make(param);

}

lssvm_objective::~lssvm_objective()
{
}



void lssvm_objective::set_dataset(const shared_ptr<dataset> & data_set_)
{

	data_set = data_set_;



}

#include <liblearning/util/matrix_util.h>

#include <limits>
#include <cmath>

NumericType lssvm_objective::prepared_value(deep_auto_encoder & net) 
{
	
	const MatrixType & feature = net.get_layered_output(net.get_coder_layer_id());
	
    svm.train ( data_set->clone_update_data(feature) );
	
	current_obj_val = svm.get_object_value();
	
	return current_obj_val;

}


vector<shared_ptr<MatrixType>> lssvm_objective::prepared_value_delta(deep_auto_encoder & net) 
{
	vector<shared_ptr<MatrixType>> result(2);

	result[0] = shared_ptr<MatrixType>(new MatrixType(svm.diffObject2Sample())) ;
	result[1] = shared_ptr<MatrixType>();
	return result;
}


lssvm_objective * lssvm_objective::clone()
{
	return new lssvm_objective(*this);
}

string lssvm_objective::get_info()
{
	string info = "LSSVM Objective Value = " + boost::lexical_cast<string>(current_obj_val);


	return info;

}