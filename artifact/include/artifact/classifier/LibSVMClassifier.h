#ifndef LIBSVM_CLASSIFIER_H
#define LIBSVM_CLASSIFIER_H

#include <liblearning/core/classifier.h>
#include <liblearning/core/maker.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/kernel/kernel.h>
#include <liblearning/classifier/svm.h>


namespace classification
{
	using namespace core;
	using namespace kernelmethod;

	struct LibSVMParam
	{
		shared_ptr<kernel> kernelfunc;
		double C;
	};


	class LibSVMClassifier : public classifier, public maker<LibSVMParam>
	{

	private:
		
		shared_ptr<supervised_dataset> train_set;

		svm_parameter * libsvm_param;

		svm_model * libsvm_machine;

		svm_problem * libsvm_problem;
		
	protected:
		
		void clear();

	public:
		LibSVMClassifier( );
		~LibSVMClassifier(void);

		virtual void make(const LibSVMParam & param);

		virtual void train(const shared_ptr<dataset>  &);

		virtual NumericType test(const shared_ptr<dataset>  &, const shared_ptr<dataset>  &);
	};

}

#endif
