#ifndef SUPERVISED_DATASUBSET_H
#define SUPERVISED_DATASUBSET_H

namespace LibLearning
{
  	namespace Core
  	{
		class SupervisedDataSubSet: public ISupervisedDataSet
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
			
			  virtual int GetSampleLabel(int i) = 0 const;
			virtual vectori<int> GetAllSampleLabel() = 0 const;
			  virtual  vector<int>  GetClassLables() = 0 const;
				virtual int GetClassNum() = 0 const;
				virtual int GetClassMemberNum(int classLable) = 0 const;
		};
	}
}

#endif