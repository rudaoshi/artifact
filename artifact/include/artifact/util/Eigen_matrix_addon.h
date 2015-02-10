friend class boost::serialization::access;


template<class Archive>
void save(Archive & ar, const unsigned int version) const
{
	boost::serialization::collection_size_type row(this->rows());
	boost::serialization::collection_size_type col(this->cols());

	ar & boost::serialization::make_nvp("row",row);
	ar & boost::serialization::make_nvp("col",col);

	ar & boost::serialization::make_nvp("elements",boost::serialization::make_array(this->data(), row*col));
}
template<class Archive>
void load(Archive & ar, const unsigned int version)
{
	boost::serialization::collection_size_type row(0);
	boost::serialization::collection_size_type col(0);

	ar & boost::serialization::make_nvp("row",row);
	ar & boost::serialization::make_nvp("col",col);

	this->resize(row,col);

	ar & boost::serialization::make_nvp("elements",boost::serialization::make_array(this->data(), row*col));

}
BOOST_SERIALIZATION_SPLIT_MEMBER()

