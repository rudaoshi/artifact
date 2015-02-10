#include <liblearning/classifier/LibSVMClassifier.h>

#include <liblearning/kernel/linear_kernel.h>

using namespace kernelmethod;

using namespace classification;

LibSVMClassifier::LibSVMClassifier():libsvm_machine(0),libsvm_problem(0),libsvm_param(0)
{
}


LibSVMClassifier::~LibSVMClassifier(void)
{
	clear();
}



void LibSVMClassifier::make(const LibSVMParam & param_)
{
	param =  param_;

	clear();

	// Default value for libsvm_param, copied from libsvm

	// default values
	libsvm_param = new svm_parameter;
	libsvm_param->svm_type = C_SVC;
	libsvm_param->kernel_type = RBF;
	libsvm_param->degree = 3;
	libsvm_param->gamma = 0;	// 1/num_features
	libsvm_param->coef0 = 0;
	libsvm_param->nu = 0.5;
	libsvm_param->cache_size = 100;
	libsvm_param->C = 1;
	libsvm_param->eps = 1e-3;
	libsvm_param->p = 0.1;
	libsvm_param->shrinking = 1;
	libsvm_param->probability = 0;
	libsvm_param->nr_weight = 0;
	libsvm_param->weight_label = NULL;
	libsvm_param->weight = NULL;

	// Set Specific Parameter 
	libsvm_param->svm_type = C_SVC;
	libsvm_param->C = param.C;

	if (dynamic_cast<linear_kernel *>(param.kernelfunc.get()))
	{
		libsvm_param->kernel_type = LINEAR;
	}


}

void LibSVMClassifier::clear()
{
	if (libsvm_machine != 0)
	{
		svm_free_and_destroy_model(&libsvm_machine);

		libsvm_machine = 0;
	}
	if (libsvm_problem != 0)
	{
		delete libsvm_problem->y;

		for (int i = 0; i < libsvm_problem->l;i++)
		{
			delete libsvm_problem->x[i];
		}

		delete libsvm_problem->x;

		delete libsvm_problem;

		libsvm_problem = 0;
	}

	if (libsvm_param != 0)
	{
		svm_destroy_param(libsvm_param);

		delete libsvm_param;

		libsvm_param = 0;
	}
}

void LibSVMClassifier::train(const shared_ptr<dataset>  & train_)
{


	train_set = dynamic_pointer_cast<supervised_dataset>( train_);

	if (!train_set)
	{
		throw new runtime_error("SVM can only deal with supervised problems!");
	}


	libsvm_problem = new svm_problem;

	libsvm_problem->l = train_set->get_sample_num();
	libsvm_problem->y = new double [libsvm_problem->l];
	libsvm_problem->x = new svm_node * [libsvm_problem->l];

	const MatrixType & data = train_set->get_data();

	EigenMatrixType temp_data = (EigenMatrixType)data;
	for (int i = 0; i < train_set->get_sample_num();i++)
	{
		libsvm_problem->x[i] = new svm_node[train_set->get_dim()+1];
		for (int j = 0; j < train_set->get_dim();j++)
		{
			libsvm_problem->x[i][j].index = j+1;
			libsvm_problem->x[i][j].value = temp_data(j,i);
		}

		libsvm_problem->x[i][train_set->get_dim()].index = -1;
		libsvm_problem->x[i][train_set->get_dim()].value = -1;

		libsvm_problem->y[i] = train_set->get_label()[i];
	}

	libsvm_machine = svm_train(libsvm_problem,libsvm_param);

}

//const VectorType & LibSVMClassifier::get_alpha()
//{
//	return alpha;
//}
//
//const VectorType & LibSVMClassifier::get_w()
//{
//	return w;
//}
//
//double LibSVMClassifier::get_b()
//{
//	return b;
//}
//
//
//double LibSVMClassifier::get_object_value()
//{
//	return 0.5*(w.squaredNorm() + alpha.squaredNorm()/param->gamma);
//}

NumericType LibSVMClassifier::test(const shared_ptr<dataset>  & , const shared_ptr<dataset>  & test_)
{


	shared_ptr<supervised_dataset>  test_set = dynamic_pointer_cast<supervised_dataset >( test_);

	if (!test_set)
	{
		throw new runtime_error("SVM can only deal with supervised problems!");
	}

	std::vector<int> test_label(test_set->get_sample_num());
	const MatrixType & data = test_set->get_data();
	EigenMatrixType temp_data = (EigenMatrixType)data;
	for (int i = 0; i < test_set->get_sample_num();i++)
	{
		svm_node * x  = new svm_node[test_set->get_dim()+1];
		for (int j = 0; j < test_set->get_dim();j++)
		{
			x[j].index = j+1;
			x[j].value = temp_data(j,i);
		}
		x[train_set->get_dim()].index = -1;
		x[train_set->get_dim()].value = -1;

		test_label[i] = svm_predict(libsvm_machine,x);

		delete x;
	}

	double correctNum = 0;

	for (int i = 0;i<test_set->get_sample_num();i++)
	{
		if (test_set->get_label()[i] == test_label[i])
		{
			correctNum ++;
		}
	}

	return correctNum / test_set->get_sample_num();

}

//MatrixType LibSVMClassifier::diffObject2Sample()
//{
//	return w*(alpha.array()*y.array()).matrix().transpose();
//}
