#ifndef KFDA_H

#define KFDA_H


#include <liblearning/core/config.h>
#include <liblearning/core/feature_extractor.h>
#include <liblearning/core/maker.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/kernel/kernel.h>

namespace subspace
{
	using namespace core;
	using namespace kernelmethod;
	struct KFDAParam
	{
		shared_ptr<kernel> kernelfunc;
		double reg;
	};

	class KFDA: public feature_extractor, public maker<KFDAParam>
	{
		shared_ptr<supervised_dataset> train_set;
		

		MatrixType eigenvector;


	public:

		KFDA ();

		virtual void train(const shared_ptr<dataset>  &);

		virtual shared_ptr<dataset> extract_feature(const shared_ptr<dataset>  & data);
	};
}

#endif