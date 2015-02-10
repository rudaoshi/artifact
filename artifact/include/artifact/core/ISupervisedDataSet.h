#ifndef ISUPERVISEDDATASET_H
#define ISUPERVISEDDATASET_H

#include "IDataSet.h"

namespace artifact
{
  namespace Core
  {
    class ISupervisedDataSet:virtual IDataSet
    {
    public:
      virtual int GetSampleLabel(int i) = 0 const;
	virtual vectori<int> GetAllSampleLabel() = 0 const;
	  virtual  vector<int>  GetClassLables() = 0 const;
		virtual int GetClassNum() = 0 const;
		virtual int GetClassMemberNum(int classLable) = 0 const;
    };
  }
}


#endif
