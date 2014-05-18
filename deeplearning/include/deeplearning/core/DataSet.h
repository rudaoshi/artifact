/*
 * dataset.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef DATASET_H_
#define DATASET_H_

#include "config.h"

//#include "data_splitter.h"

#include "serialize.h"


#include <vector>
#include <string>
#include <memory>

using namespace std;

#include <liblearning/Core/IDataSet.h>

#include <boost/serialization/vector.hpp> 
#include <rapidxml/rapidxml.hpp>

#include "identifiable.h"

namespace LibLearning
{
  namespace Core
  {
	class dataset_group;
	class POCO_EXPORT DataSet : public IDataSet//,public identifiable
	{ 

	protected:

		MatrixType data;

	protected:

	
		DataSet();

	public:

		DataSet();

		DataSet(const MatrixType & data_);

		DataSet(const DataSet & data_set);


		virtual ~DataSet();

      	virtual VectorType  GetSample(int i ) const;
		virtual VectorType &  GetSample(int i ) = 0;
      	virtual MatrixType  GetSampleSet() const;

      	virtual unsigned int GetSampleDim() const;
      	virtual unsigned int GetSampleNum() const;

		virtual shared_ptr<IDataSet> operator + (const IDataSet & ) = 0 const;

		virtual shared_ptr<IDataSet> clone() = 0 const;
//		virtual dataset_group split(const dataset_splitter & maker) const;

		virtual void append(const dataset & data_set);

		virtual void copy(const dataset & data_set);

		virtual shared_ptr<dataset> clone() const;

		virtual shared_ptr<dataset> clone_update_data(const MatrixType & data) const;

		virtual shared_ptr<dataset> sub_set(const vector<int> & index) const;

	public:

		// Plain XML serialization



		virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

		virtual void decode_xml_node(rapidxml::xml_node<> & node);



		// HDF5 serialization

		virtual void encode_hdf_node(H5::Group * obj) const;

		virtual void decode_hdf_node(const H5::Group * obj) ;



	};

}


}

CAMP_TYPE(core::dataset);

#endif /* DATASET_H_ */
