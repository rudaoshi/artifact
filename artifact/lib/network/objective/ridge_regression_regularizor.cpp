#include <liblearning/deep/objective/ridge_regression_regularizor.h>

#include <liblearning/deep/deep_auto_encoder.h>
using namespace deep;
using namespace deep::objective;
ridge_regression_regularizor::ridge_regression_regularizor(void)
{
}


ridge_regression_regularizor::~ridge_regression_regularizor(void)
{
}


NumericType ridge_regression_regularizor::value(deep_auto_encoder & net)
{
	cur_obj_val = net.get_Wb().squaredNorm();
	return cur_obj_val;
}
	
tuple<NumericType, VectorType> ridge_regression_regularizor::value_diff(deep_auto_encoder & net)
{
	NumericType value = cur_obj_val;
	VectorType value_diff = (VectorType)(2*net.get_dWb());

	return  make_tuple(value,value_diff);
}


ridge_regression_regularizor * ridge_regression_regularizor::clone()
{
	return new ridge_regression_regularizor(*this);
}

string ridge_regression_regularizor::get_info()
{
	return "WDecay Objective = " +  boost::lexical_cast<string>(cur_obj_val);
}