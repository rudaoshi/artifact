#ifndef MAX_MARGIN_OBJECTIVE_H_
#define MAX_MARGIN_OBJECTIVE_H_


#include "data_related_network_objective.h"

#include <liblearning/classifier/LeastSquareSVM.h>

namespace deep
{
	namespace objective
	{
		using namespace classification;
		class POCO_EXPORT MaximumMarginObjective: public data_related_network_objective
		{

			LeastSquareSVM svm;

			double gamma;

		public:
			MaximumMarginObjective(double gamma);
			~MaximumMarginObjective();


			virtual NumericType prepared_value(deep_auto_encoder & net) ;
			virtual vector<shared_ptr<MatrixType>> prepared_value_delta(deep_auto_encoder & net) ;

			virtual MaximumMarginObjective * clone();

			virtual string get_info();
		};
	}
}

#endif