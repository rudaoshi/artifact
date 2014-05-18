#ifndef CLONALBE_OBJECT_H
#define CLONALBE_OBJECT_H


#include <memory>
#include <type_traits>


#include <loki/Factory.h>

namespace core
{
	template <typename T, bool b>
	T*  do_clone(T * ptr, const std::integral_constant<bool, b> &)
	{
		return 0;
	}



	template <typename T>
	T* do_clone(T * ptr, const std::false_type&)
	{
		return  new T(*ptr);
	}




	template <typename  T>
	class clonable
	{

	protected:
		virtual T* clone ()
		{

			typedef std::integral_constant<bool, std::is_abstract<T>::value > truth_type;

			return do_clone(dynamic_cast<T*>(this), truth_type());
		}
	};

}

#endif
