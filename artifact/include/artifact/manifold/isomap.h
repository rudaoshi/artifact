#ifndef ISOMAP_H
#define ISOMAP_H

#include <liblearning/core/dataset.h>

namespace manifold
{
	using namespace core;

	class isomap
	{
		const dataset & data;
		MatrixType features;
	public:
		isomap(const dataset & data_, int max_component);
		~isomap(void);
	};
}
#endif
