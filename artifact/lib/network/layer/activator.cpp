
#include <artifact/network/layer/activator.h>

using namespace artifact::network;


RowVectorType linear_activator::activate(const RowVectorType & v)
{
    return v;
}


RowVectorType linear_activator::gradient(const RowVectorType & v)
{
    return VectorType::Ones(v.size());
}


RowVectorType linear_activator::gradient(const RowVectorType & v, const RowVectorType & activated)
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

MatrixType linear_activator::gradient(const MatrixType & m, const MatrixType & activated)
{
    return MatrixType::Ones(m.rows(), m.cols());
}
RowVectorType logistic_activator::activate(const RowVectorType & v)
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



RowVectorType logistic_activator::gradient(const RowVectorType & v)
{
    throw runtime_error("This should not be used in optimization");
}

RowVectorType logistic_activator::gradient(const RowVectorType & m,
        const RowVectorType & activated)
{
    return activated.array() * (1 - activated.array());
}

MatrixType logistic_activator::gradient(const MatrixType & m)
{
    throw runtime_error("This should not be used in optimization");
}

MatrixType logistic_activator::gradient(const MatrixType & m,
        const MatrixType & activated)
{
    return activated.array() * (1 - activated.array());
}

RowVectorType logistic_activator::activate(const RowVectorType & v)
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



RowVectorType relu_activator::gradient(const RowVectorType & v)
{
    return (1 + (-v).array().exp()).inverse();
}

RowVectorType relu_activator::gradient(const RowVectorType & v,
                                           const RowVectorType & activated)
{
    return (1 + (-v).array().exp()).inverse();
}

MatrixType relu_activator::gradient(const MatrixType & m)
{
    return (1 + (-m).array().exp()).inverse();
}

MatrixType relu_activator::gradient(const MatrixType & m,
                                        const MatrixType & activated)
{
    return (1 + (-m).array().exp()).inverse();
}

RowVectorType relu_activator::activate(const RowVectorType & v)
{
#ifdef USE_GPU

	return m.array().logistic();

#elif defined USE_PARTIAL_GPU

	return m.array().logistic();

#else

    return (1 + v.array().exp()).log();

#endif
}


MatrixType relu_activator::activate(const MatrixType & m)
{
#ifdef USE_GPU

	return m.array().logistic();

#elif defined USE_PARTIAL_GPU

	return m.array().logistic();

#else

    return (1 + m.array().exp()).log();

#endif
}

