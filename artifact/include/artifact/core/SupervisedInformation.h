#ifndef SUPERVISED_INFORMATION_H
#define SUPERVISED_INFORMATION_H


namespace artifact
{
  namespace Core
  {
    class ISupervisedDataSet:virtual IDataSet
    {
    public:
      virtual int GetSampleLabel(int i) = 0 const;
    };
  }
}


#endif