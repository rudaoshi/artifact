
#include <artifact/network/layer/activator.h>

using namespace artifact::network;


VectorType linear_activator::activate(const VectorType & v)
{
    return v;
}


VectorType linear_activator::gradient(const VectorType & v)
{
    return VectorType::Ones(v.size());
}

MatrixType linear_activator::activate(const MatrixType & v)
{
    return v;
}


MatrixType linear_activator::gradient(const MatrixType & m)
{
    return MatrixType::Ones(m.rows(), m.cols());
}


VectorType logistic_activator::activate(const VectorType & v)
{
#ifdef USE_GPU

	return m.array().logistic();

#elif defined USE_PARTIAL_GPU

	return m.array().logistic();

#else

    return (1 + (-v).array().exp()).inverse();

#endif
}


MatrixType logistic_activator::activate(const MatrixType & m)
{
#ifdef USE_GPU

	return m.array().logistic();

#elif defined USE_PARTIAL_GPU

	return m.array().logistic();

#else

    return (1 + (-m).array().exp()).inverse();

#endif
}



VectorType logistic_activator::gradient(const VectorType & v)
{
    return  v.array() * (1 - v.array());
}

MatrixType logistic_activator::gradient(const MatrixType & m)
{
    return  m.array() * (1 - m.array());
}