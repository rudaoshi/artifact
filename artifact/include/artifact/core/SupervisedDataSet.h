
#ifndef SUPERVISED_DATASET_H_
#define SUPERVISED_DATASET_H_

#include "ISupervisedDataSet.h"
#include <vector>

using namespace std;

namespace artifact
{
  namespace Core
  {
	class  SupervisedDataSet : public ISupervisedDataSet
	{

	protected:


		MatrixType data;
		vector<int> label;
		vector<int> class_id;
		vector<int> class_elem_num;

		void calculate_supervised_info();

	public:
		SupervisedDataSet();
		SupervisedDataSet(const MatrixType & data, const vector<int> & label);
		SupervisedDataSet(const SupervisedDataSet & data_set);

		~SupervisedDataSet(void);

	      virtual int GetSampleLabel(int i) const;
		virtual vectori<int> GetAllSampleLabel()  const;
		  virtual  vector<int>  GetClassLables() const;
			virtual int GetClassNum()const;
			virtual int GetClassMemberNum(int classLable) const;


		virtual void copy(const dataset & data_set);

		virtual shared_ptr<dataset> clone() const;

		virtual shared_ptr<dataset> clone_update_data(const MatrixType & data) const;

		virtual shared_ptr<dataset> sub_set(const vector<int> & index) const;

	public:

		// Plain XML serialization



		virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

		virtual void decode_xml_node(rapidxml::xml_node<> & node);



		// HDF5 serialization

		virtual void encode_hdf_node(H5::Group * group) const;

		virtual void decode_hdf_node(const H5::Group * obj) ;



	};

	// CAMP_TYPE(supervised_dataset)

}

}
CAMP_TYPE(core::supervised_dataset);

#endif /*SUPERVISED_DATASET_H_*/