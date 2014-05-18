#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <liblearning/core/dataset.h>
#include <liblearning/core/classifier.h>

namespace core
{

	template<typename FE, typename CL> struct recognition_system_param
	{
		shared_ptr<typename FE::ParamType> f_param;
		shared_ptr<typename CL::ParamType> c_param;
	};


	template <typename FE, typename CL> class recognition_system: public classifier<recognition_system_param<FE,CL>>
	{
		shared_ptr< recognition_system_param<FE,CL> > param_;
		FE feature_extractor_;
		CL classifier_;

	public:

		virtual void make(const shared_ptr<recognition_system_param<FE,CL>> & param)
		{
			param_ = param;

			feature_extractor_.make(param_->f_param);
			classifier_.make(param_->c_param);
		}

		virtual void train(const shared_ptr<dataset> & traindata)
		{
			feature_extractor_.train(traindata);
			shared_ptr<dataset> feature = feature_extractor_.extract_feature(traindata);
			classifier_.train(feature);
		}

		virtual NumericType test(const shared_ptr<dataset> & traindata,const shared_ptr<dataset> & testdata)
		{
			shared_ptr<dataset> trainfeature = feature_extractor_.extract_feature(traindata); 

			shared_ptr<dataset> testfeature = feature_extractor_.extract_feature(testdata);

			return classifier_.test(trainfeature, testfeature);
		}

	};
}

#endif