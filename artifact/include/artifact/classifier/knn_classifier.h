#ifndef KNN_CLASSIFIER_H_
#define KNN_CLASSIFIER_H_

#include <liblearning/core/supervised_dataset.h>

namespace classification
{
	using namespace core;
	class knn_classifier
	{
		const supervised_dataset & train;
		int k;
	public:
		knn_classifier(const supervised_dataset &, int );
		~knn_classifier(void);

		NumericType test(const supervised_dataset &);
	};
}
#endif
