#ifndef LDA_H

#define LDA_H


#include <liblearning/core/config.h>
#include <liblearning/core/feature_extractor.h>
#include <liblearning/core/maker.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/kernel/kernel.h>

namespace subspace
{
	using namespace core;
	using namespace kernelmethod;
	struct LDAParam
	{
		double reg;
	};

	class LDA: public feature_extractor, public maker<LDAParam>
	{
		shared_ptr<supervised_dataset> train_set;
		
		shared_ptr<LDAParam> param;

		MatrixType eigenvector;


	public:

		LDA ();

		virtual void train(const shared_ptr<dataset>  &);

		virtual shared_ptr<dataset> extract_feature(const shared_ptr<dataset>  & data);
	};
}

#endif