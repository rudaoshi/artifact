#ifndef ALGO_UTIL_H_
#define ALGO_UTIL_H_


#include <algorithm>
#include <iterator>
#include <vector>
#include <boost/iterator/counting_iterator.hpp>

using namespace std;
template < typename RandomAccessIterator >
class SortHelper
{
	RandomAccessIterator & first;
public :
	SortHelper(RandomAccessIterator & first_):first(first_)
	{
	}

	bool operator()(unsigned int a, unsigned int b)
	{
		return * (first + a) < * (first + b);
	}
};

template <class RandomAccessIterator>
vector<unsigned int> sort_index(RandomAccessIterator first, RandomAccessIterator last)
{
	int N = last-first;
	vector<unsigned int> index(N);

	std::copy(
		boost::counting_iterator<unsigned int>(0),
		boost::counting_iterator<unsigned int>(N), 
		std::back_inserter(index));

	//std::sort(index.begin(),index.end(),
	//	[&](const unsigned int a, const unsigned int b) ->bool 
	//	{ 
	//		return * (first + a) < * (first + b);
	//	}
	//);

	std::sort(index.begin(),index.end(),SortHelper<RandomAccessIterator>(first));
	return index;

}

template <class RandomAccessIterator>
vector<unsigned int> nth_element_index(
	RandomAccessIterator first, 
	RandomAccessIterator middle, 
	RandomAccessIterator last
	)
{
	int N = last-first;
	int nth_pos = middle - first;
	vector<unsigned int> index(N);

	std::copy(
		boost::counting_iterator<unsigned int>(0),
		boost::counting_iterator<unsigned int>(N), 
		std::back_inserter(index));

	//std::nth_element(index.begin(),index.begin()+nth_pos,index.end(),
	//	[&](const unsigned int a, const unsigned int b) ->bool 
	//	{ 
	//		return * (first + a) < * (first + b);
	//	}
	//);
	std::nth_element(index.begin(),index.begin()+nth_pos,index.end(),SortHelper<RandomAccessIterator>(first));
	index.erase(index.begin()+nth_pos,index.end());

	return index;

}

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

class StringEmptyPredicator
{
public:
	bool operator ()(const string & e)
	{
		return e.empty();
	}

};

template <typename T, typename SplitPredicate>
vector<T> construct_array(const string & coded_str, const SplitPredicate  & p)
{
	typedef vector< string > split_vector_type;

	split_vector_type v;
	boost::split( v, coded_str, p);
	
	//auto v_end = remove_if(v.begin(),v.end(),[](const string & e)-> bool { return e.empty(); });

	split_vector_type::iterator v_end = remove_if(v.begin(),v.end(),StringEmptyPredicator());

	int num = v_end - v.begin();

	vector<T > t_v(num);

	for (int i = 0;i < num ;i++)
	{
		t_v[i] = boost::lexical_cast<T>(v[i]);
	}

	return t_v;

}



#endif
