#ifndef SAMPLE_STORAGE_H
#define SAMPLE_STORAGE_H

namespace LibLearning
{
  	namespace Core
  	{
		class VectorSampleStorage
		{
			protected:
				// each column represents a sample
				MatrixType sample;
				
			public:
				VectorSampleStorage(const MatrixType & sample_);
				virtual ~VectorSampleStorage();
				
				MatrixType & GetSampleSet();
				VectorType & GetSample(unsigned int i);
			};
		
	}
}
#endif