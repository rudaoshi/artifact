#include <liblearning/deep/objective/MaximumMarginObjective.h>



#include <liblearning/util/matrix_util.h>




#include <liblearning/deep/deep_auto_encoder.h>
#include <liblearning/deep/neuron_layer_operation.h>
#include <liblearning/kernel/linear_kernel.h>
using namespace deep;
using namespace deep::objective;
using namespace kernelmethod;



MaximumMarginObjective::MaximumMarginObjective(double gamma_)
{
	type = encoder_related;
	gamma = gamma_;
}


MaximumMarginObjective::~MaximumMarginObjective(void)
{
}



NumericType MaximumMarginObjective::prepared_value(deep_auto_encoder & net) 
{
	const MatrixType & feature = net.get_layered_output(net.get_coder_layer_id());

	const supervised_dataset & sdata_set = dynamic_cast<const supervised_dataset &>(*data_set);
	shared_ptr<supervised_dataset> feature_set(new supervised_dataset(feature,sdata_set.get_label()));
	linear_kernel linearKern; 

	LeastSquareSVMParam param;
	param.gamma = gamma;
	param.kernelfunc.reset(new linear_kernel());
	svm.make(param);
	svm.train(feature_set);

	return svm.get_object_value();

}

vector<shared_ptr<MatrixType>> MaximumMarginObjective::prepared_value_delta(deep_auto_encoder & net) 
{                                                                                                                                      
	const MatrixType & feature = net.get_layered_output(net.get_coder_layer_id());

	NumericType N = data_set->get_sample_num();

	shared_ptr<MatrixType> obj_diff( new MatrixType( svm.diffObject2Sample()));

	vector<shared_ptr<MatrixType>> result(2);

	result[0] = obj_diff;
	result[1] = shared_ptr<MatrixType>();

	return result;
}


MaximumMarginObjective * MaximumMarginObjective::clone()
{
	return new MaximumMarginObjective(*this);
}

string MaximumMarginObjective::get_info()
{
	return "";
}
