#ifndef IDATASET_H
#define IDATASET_H


namespace artifact
{
  namespace Core
  {
    class IDataSet: public direct_xml_file_serializable, public direct_hdf_file_serializable
    {
    public:
      virtual VectorType  GetSample(int i ) = 0 const;
	  virtual VectorType &  GetSample(int i ) = 0;
      virtual MatrixType  GetSampleSet() = 0 const;

      virtual unsigned int GetSampleDim() = 0 const;
      virtual unsigned int GetSampleNum() = 0 const;

	  virtual shared_ptr<IDataSet> operator + (const IDataSet & ) = 0 const;

	virtual shared_ptr<IDataSet> clone() = 0 const;
    };


  }
}

#endif
