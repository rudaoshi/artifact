
#ifndef DATA_SET_H_
#define DATA_SET_H_

#include <Eigne/Core>

namespace deeplearning
{
  template <typename SampleType>
  class sample_set_type
  {

  };

  template<>
  class sample_set_type<Eigen::VectorXd>
  {
  public:
    typedef Eigen::MatrixXd type;
  }

  template<>
  class sample_set_type<Eigen::VectorXf>
  {
  public:
    typedef Eigen::MatrixXf type;
  }

}
#endif
