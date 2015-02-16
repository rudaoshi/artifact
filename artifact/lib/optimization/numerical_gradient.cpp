

#include <boost/math/constants/constants.hpp>
#include <cmath>

template<typename value_type, typename function_type>
value_type derivative(const value_type x, const value_type dx, function_type func)
{
    // Compute d/dx[func(*first)] using a three-point
    // central difference rule of O(dx^6).

    const value_type dx1 = dx;
    const value_type dx2 = dx1 * 2;
    const value_type dx3 = dx1 * 3;

    const value_type m1 = (func(x + dx1) - func(x - dx1)) / 2;
    const value_type m2 = (func(x + dx2) - func(x - dx2)) / 4;
    const value_type m3 = (func(x + dx3) - func(x - dx3)) / 6;

    const value_type fifteen_m1 = 15 * m1;
    const value_type six_m2     =  6 * m2;
    const value_type ten_dx1    = 10 * dx1;

    return ((fifteen_m1 - six_m2) + m3) / ten_dx1;
}

#include <artifact/optimization/numerical_gradient.h>

#include <iostream>

VectorType artifact::optimization::numerical_gradient(optimizable & machine, const VectorType & param,  const MatrixType & X, const VectorType & y)
{

    VectorType dp = VectorType::Zero(param.size());

    NumericType dx = 0.001; //std::pow(std::numeric_limits<NumericType>::epsilon(), 1.0/3.0);

    for (int i = 0; i < param.size(); i++)
    {
        dp(i) = derivative(param[i], dx,
                [&](NumericType x) -> NumericType
                {
                    VectorType p = param;
                    p[i] = x;
                    machine.set_parameter(p);
                    return machine.objective(X, y);
                } );

//        std::cerr << i << std::endl;
    }

    return dp;


}