#include <liblearning/classifier/LinearSVMClassifier.h>



using namespace classification;

LinearSVMClassifier::LinearSVMClassifier():linear_svm_machine(0),linear_svm_problem(0),linear_svm_param(0)
{
}


LinearSVMClassifier::~LinearSVMClassifier(void)
{
	clear();
}



void LinearSVMClassifier::make(const LinearSVMParam & param_)
{
	param =  param_;

	clear();

	// Default value for libsvm_param, copied from libsvm

	// default values
	linear_svm_param = new parameter;
	linear_svm_param->solver_type = L2R_L2LOSS_SVC_DUAL;
	linear_svm_param->C = 1;
	linear_svm_param->eps = std::numeric_limits<double>::max(); // see setting below
	linear_svm_param->nr_weight = 0;
	linear_svm_param->weight_label = NULL;
	linear_svm_param->weight = NULL;




	// Set Specific Parameter 

	linear_svm_param->C = param.C;

	linear_svm_param->solver_type = L2R_L1LOSS_SVC_DUAL;  
//	linear_svm_param->solver_type = L2R_L2LOSS_SVC;  // faster when the number of instance is much larger than that of dimensionality

	if(linear_svm_param->eps == std::numeric_limits<double>::max())
	{
		if(linear_svm_param->solver_type == L2R_LR || linear_svm_param->solver_type == L2R_L2LOSS_SVC)
			linear_svm_param->eps = 0.01;
		else if(linear_svm_param->solver_type == L2R_L2LOSS_SVC_DUAL || linear_svm_param->solver_type == L2R_L1LOSS_SVC_DUAL || linear_svm_param->solver_type == MCSVM_CS || linear_svm_param->solver_type == L2R_LR_DUAL)
			linear_svm_param->eps = 0.1;
		else if(linear_svm_param->solver_type == L1R_L2LOSS_SVC || linear_svm_param->solver_type == L1R_LR)
			linear_svm_param->eps = 0.01;
	}

	

	//if (dynamic_cast<linear_kernel *>(param->kernelfunc.get()))
	//{
	//	libsvm_param->kernel_type = LINEAR;
	//}


}

void LinearSVMClassifier::clear()
{
	if (linear_svm_machine != 0)
	{
		free_and_destroy_model(&linear_svm_machine);

		linear_svm_machine = 0;
	}
	if (linear_svm_problem != 0)
	{
		delete linear_svm_problem->y;

		for (int i = 0; i < linear_svm_problem->l;i++)
		{
			delete linear_svm_problem->x[i];
		}

		delete linear_svm_problem->x;

		delete linear_svm_problem;

		linear_svm_problem = 0;
	}

	if (linear_svm_param != 0)
	{
		destroy_param(linear_svm_param);

		delete linear_svm_param;

		linear_svm_param = 0;
	}
}

void LinearSVMClassifier::train(const shared_ptr<dataset>  & train_)
{


	train_set = dynamic_pointer_cast<supervised_dataset>( train_);

	if (!train_set)
	{
		throw new runtime_error("SVM can only deal with supervised problems!");
	}


	linear_svm_problem = new problem;

	linear_svm_problem->l = train_set->get_sample_num();
	linear_svm_problem->n = train_set->get_dim();

	linear_svm_problem->y = new int [linear_svm_problem->l];
	linear_svm_problem->x = new feature_node * [linear_svm_problem->l];

	linear_svm_problem->bias = -1;

	const MatrixType & data = train_set->get_data();

	EigenMatrixType temp_data = (EigenMatrixType)data;
	for (int i = 0; i < train_set->get_sample_num();i++)
	{
		linear_svm_problem->x[i] = new feature_node[train_set->get_dim()+1];
		for (int j = 0; j < train_set->get_dim();j++)
		{
			linear_svm_problem->x[i][j].index = j+1;
			linear_svm_problem->x[i][j].value = temp_data(j,i);
		}

		linear_svm_problem->x[i][train_set->get_dim()].index = -1;
		linear_svm_problem->x[i][train_set->get_dim()].value = -1;

		linear_svm_problem->y[i] = train_set->get_label()[i];
	}

	linear_svm_machine = ::train(linear_svm_problem,linear_svm_param);

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

NumericType LinearSVMClassifier::test(const shared_ptr<dataset>  & , const shared_ptr<dataset>  & test_)
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
		feature_node * x  = new feature_node[test_set->get_dim()+1];
		for (int j = 0; j < test_set->get_dim();j++)
		{
			x[j].index = j+1;
			x[j].value = temp_data(j,i);
		}
		x[train_set->get_dim()].index = -1;
		x[train_set->get_dim()].value = -1;

		test_label[i] = predict(linear_svm_machine,x);

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
