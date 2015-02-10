#include <Eigen/LU>
#include <liblearning/classifier/LeastSquareSVM.h>


using namespace classification;

LeastSquareSVM::LeastSquareSVM()
{
	
}


LeastSquareSVM::~LeastSquareSVM(void)
{
}


void LeastSquareSVM::train(const shared_ptr<dataset>  & train_)
{
	train_set = dynamic_pointer_cast<supervised_dataset>( train_);

	if (!train_set)
	{
		throw new runtime_error("SVM can only deal with supervised problems!");
	}

	if(train_set->get_class_num() != 2)
	{
		throw new runtime_error("SVM can only deal with 2 class problems!");
	}

	y.resize(train_set->get_sample_num());

	for (int i = 0;i<train_set->get_sample_num();i++)
	{
		if(train_set->get_label()[i] == train_set->get_class_id()[0])
		{
			y(i) = 1.0;
		}
		else
		{
			y(i) = -1.0;
		}
	}

	int N = train_set->get_sample_num();

	MatrixType omega = y.asDiagonal()*param.kernelfunc->eval(train_set->get_data(),train_set->get_data())*y.asDiagonal();

	//for (int i = 0;i<N;i++)
	//{
	//	omega.row(i).array() *= y.array();
	//}

	//for (int i = 0;i<N;i++)
	//{
	//	omega.col(i).array() *= y.array();
	//}

	MatrixType A(N+1,N+1);

	A(0,0) = 0;
	A.block(1,0,N,1) = y;
	A.block(0,1,1,N) = y.transpose();
	A.block(1,1,N,N) = omega;

	VectorType zo(N+1);

	zo(0) = 0;
	zo.tail(N).fill(1);

	A += (zo/param.gamma).asDiagonal();

	VectorType alpha_b = A.fullPivLu().solve(zo);

	b = alpha_b(0);

	alpha = alpha_b.tail(N);

	w = train_set->get_data()*(alpha.array()*y.array()).matrix();

}

const VectorType & LeastSquareSVM::get_alpha()
{
	return alpha;
}

const VectorType & LeastSquareSVM::get_w()
{
	return w;
}

double LeastSquareSVM::get_b()
{
	return b;
}


double LeastSquareSVM::get_object_value()
{
	return 0.5*(w.squaredNorm() + alpha.squaredNorm()/param.gamma);
}

NumericType LeastSquareSVM::test(const shared_ptr<dataset>  & , const shared_ptr<dataset>  & test_)
{
	double correctNum = 0;

	shared_ptr<supervised_dataset>  test_set = dynamic_pointer_cast<supervised_dataset >( test_);

	MatrixType gram = param.kernelfunc->eval(test_set->get_data(),train_set->get_data());

	VectorType test_y = (gram*(alpha.array()*y.array()).matrix()).array() + b;

	for (int i = 0;i<test_set->get_sample_num();i++)
	{
		if (test_set->get_label()[i] == train_set->get_class_id()[0] && test_y(i) >= 0)
		{
			correctNum ++;
		}
		else if (test_set->get_label()[i] == train_set->get_class_id()[1] && test_y(i) < 0)
		{
			correctNum ++;
		}
	}

	return correctNum / test_set->get_sample_num();

}

MatrixType LeastSquareSVM::diffObject2Sample()
{
	return w*(alpha.array()*y.array()).matrix().transpose();
}
