

#ifndef MAKER_H
#define MAKER_H

namespace core
{

	class param
	{
	};

	template<typename Param> class maker
	{
	protected:
		Param param;

		public:

			maker()
			{
			}

			maker(const maker & other)
			{
				param = other.param;
			}

			Param & get_param()
			{
				return param;
			}

			const Param & get_param() const
			{
				return param;
			}

			virtual void make(const Param & param_)
			{
				param = param_;
			}

	};

}


#endif
