#ifndef DATASUBSET_H
#define DATASUBSET_H

namespace LibLearning
{
  	namespace Core
  	{
		class DataSubSet: public IDataSet
		{
		protected:
			IDataSet * parent;
			vector<int> index;
			
			  virtual VectorType  GetSample(int i )  const;
			  virtual VectorType &  GetSample(int i ) ;
		      virtual MatrixType  GetSampleSet()  const;

		      virtual unsigned int GetSampleDim()  const;
		      virtual unsigned int GetSampleNum()  const;

			  virtual shared_ptr<IDataSet> operator + (const IDataSet & )  const;

			virtual shared_ptr<IDataSet> clone()  const;
		};
	}
}

#endif