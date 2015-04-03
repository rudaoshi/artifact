
#ifndef ARTIFACT_UTILS_BLOCK_ITERATOR_H
#define ARTIFACT_UTILS_BLOCK_ITERATOR_H

#include <algorithm>

#include <artifact/config.h>


namespace artifact
{
    namespace utils
    {
        /**
        * Mini-batch iterator of huge data set.
        *
        * Used for mini-batch optimization algorithms such as sgd;
        * For interface consitence, it also support a fake view of empty matrix (nullptr)
        */
        template<typename M>
        class batch_iterator
        {
            const M * m;
            const int block_size;

            int pos;

        public:
            batch_iterator(const M * m_, int block_size_)
                    :m(m_), block_size(block_size_), pos(0)
            {

            }

            operator bool ()
            {
                return (m != nullptr) and pos < m->rows();
            }

            M operator * ()
            {
                if (!(*this))
                {
                    throw runtime_error("Bad iterator");
                }
                int cur_block_size = std::min(block_size, int(m->rows() - pos));
                return m->block(pos, 0, cur_block_size, m->cols());
            }

            batch_iterator<M> & operator ++ ()
            {
                pos += block_size;
                return * this;
            }

            batch_iterator<M>  operator ++ (int)
            {
                batch_iterator<M> temp = *this;

                pos += block_size;
                return temp;
            }

            batch_iterator<M> & operator +=(int shift)
            {
                pos += block_size * shift;
                return * this;
            }

            batch_iterator<M> & operator --()
            {
                pos -= block_size;
                return * this;
            }

            batch_iterator<M>  operator -- (int)
            {
                batch_iterator<M> temp = *this;

                pos -= block_size;
                return temp;
            }

            batch_iterator<M> & operator -=(int shift)
            {
                pos -= block_size * shift;
                return * this;
            }

        };
    }
}

#endif