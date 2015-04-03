#include <thread>
#include <vector>
#include <iostream>

#include <artifact/optimization/mt_sgd_optimizer.h>
#include <artifact/utils/batch_iterator.h>

using namespace artifact::optimization;
using namespace artifact::utils;

mt_sgd_optimizer::mt_sgd_optimizer()
{
    this->learning_rate = 0.01;
    this->decay_rate = 0.9;
    this->thread_num = 2;
    this->batch_per_thread = 5000;
    this->max_epoches = 10;
}


mt_sgd_optimizer::~mt_sgd_optimizer(void)
{
}


void compute_gradient(MatrixType & gradients,
        gradient_optimizable & opt,
        const VectorType & param,
        batch_iterator<MatrixType> X_iter,
        batch_iterator<MatrixType> y_iter,
        int batch_id)
{
    NumericType fval;
    VectorType grad;

    opt.set_parameter(param);
    MatrixType cur_X = *X_iter;
    MatrixType cur_y;
    MatrixType * p_cur_y = 0;
    if (y_iter)
    {
        cur_y = *y_iter;
        p_cur_y = &cur_y;
    }

    tie(fval,grad) = opt.gradient(cur_X, p_cur_y);


    gradients.col(batch_id) = grad;

}


VectorType mt_sgd_optimizer::optimize(optimizable & obj,
        const VectorType & param0,
        const MatrixType & X,
        const MatrixType * y // nullptr for unsupervised optimizer
)
{


    gradient_optimizable & opt = dynamic_cast<gradient_optimizable &>(obj);

    NumericType fval = numeric_limits<NumericType>::quiet_NaN();

    VectorType grad;

    VectorType param = param0;

    NumericType cur_learning_rate = this->learning_rate;
    for (int i = 0; i < this->max_epoches; i++)
    {


        MatrixType gradients(param0.size(), this->thread_num);
        batch_iterator<MatrixType> X_iter(&X, this->batch_per_thread);
        batch_iterator<MatrixType> y_iter(y, this->batch_per_thread);

        while (X_iter)
        {
            std::vector<std::thread> threads;
            for (int j = 0; j < this->thread_num; ++j)
            {
                if (not X_iter) {
                    break;
                }

                threads.push_back(std::thread (
                        ::compute_gradient,
                        std::ref(gradients),
                        std::ref(opt),
                        std::cref(param),
                        X_iter,
                        y_iter,
                        j));

                X_iter ++;
                y_iter ++;
            }

            for(auto &t : threads){
                t.join();
            }

            grad = gradients.rowwise().sum();

            param -= cur_learning_rate*grad/grad.norm();

            X_iter ++;
            y_iter ++;
        }


        cur_learning_rate *= this->decay_rate;
    }
    return param;
}

