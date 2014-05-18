#ifndef PROTOTYPE_FACTORY_H
#define PROTOTYPE_FACTORY_H

#include <liblearning/core/config.h>

namespace core
{
	template <typename T>
	class prototype_factory
	{
	protected:

		shared_ptr<T> prototype;
	public:
		template <typename SubT>
		prototype_factory(const shared_ptr<SubT>& proto_):prototype(proto_)
		{
		}
		virtual ~prototype_factory(void)
		{
		}

		virtual shared_ptr<T> create_new()
		{
			return shared_ptr<T>(prototype->clone());
		}

	};
}

#endif