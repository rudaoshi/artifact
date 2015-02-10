#ifndef LINEAR_SVM_CLASSIFIER_H
#define LINEAR_SVM_CLASSIFIER_H

#include <liblearning/core/classifier.h>
#include <liblearning/core/maker.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/classifier/linear.h>


namespace classification
{

	using namespace core;

	struct LinearSVMParam
	{
		double C;
	};


	class LinearSVMClassifier : public classifier, public maker<LinearSVMParam>
	{

	private:

		shared_ptr<supervised_dataset> train_set;

		parameter * linear_svm_param;

		model * linear_svm_machine;

		problem * linear_svm_problem;

	protected:

		void clear();

	public:
		LinearSVMClassifier( );
		~LinearSVMClassifier(void);

		virtual void make(const LinearSVMParam & param);

		virtual void train(const shared_ptr<dataset>  &);

		virtual NumericType test(const shared_ptr<dataset>  &, const shared_ptr<dataset>  &);
	};

}

#endif
