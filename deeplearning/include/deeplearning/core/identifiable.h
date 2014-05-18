#ifndef IDENTIFIABLE_H
#define IDENTIFIABLE_H

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>


namespace core
{
	class identifiable
	{
	private:
		boost::uuids::uuid id;

	public:

		identifiable(): id(boost::uuids::random_generator()())
		{
		}

		bool operator==(identifiable const& rhs) const 
		{
			return id == rhs.id;
		}
    };

	
}

#endif