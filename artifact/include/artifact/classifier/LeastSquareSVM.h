#ifndef LEAST_SQUARE_SVM_H
#define LEAST_SQUARE_SVM_H

#include <liblearning/core/classifier.h>
#include <liblearning/core/maker.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/kernel/kernel.h>

namespace classification
{
	using namespace core;
	using namespace kernelmethod;

	struct LeastSquareSVMParam
	{
		shared_ptr<kernel> kernelfunc;
		double gamma;
	};


	class LeastSquareSVM : public classifier, public maker<LeastSquareSVMParam>
	{

	public:

		shared_ptr<supervised_dataset> train_set;

		VectorType y;

		VectorType alpha;

		VectorType w;
		double b;

	public:
		LeastSquareSVM( );
		~LeastSquareSVM(void);


		const VectorType & get_alpha();

		const VectorType & get_w();

		double get_b();

		double get_object_value();

		MatrixType diffObject2Sample();

		virtual void train(const shared_ptr<dataset>  &);

		virtual NumericType test(const shared_ptr<dataset>  &, const shared_ptr<dataset>  &);
	};

}

#endif
