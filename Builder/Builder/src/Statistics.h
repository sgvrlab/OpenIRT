#ifndef STATISTICS_H
#define STATISTICS_H

#include "Defines.h"
#include <iomanip>

template <class Value> class Stats;
template <class Value> inline STD ostream& operator << (STD ostream& os, const Stats<Value>& s);

/** incrementally accumulate statistical information */
template <class Value>
class Stats
{
public:
	/// initialize
	Stats()
	{
		init();
	}
	/// initialize and consider one value
	Stats(const Value& v)
	{
		init(v);
	}
	/// initialize and consider one value
	void init(const Value& v)
	{
		min = v;
		max = v;
		sum = v;
		sms = sqr(v);
		cnt = 1;
	}
	/// initialize
	void init()
	{
		sum = 0;
		sms = 0;
		cnt = 0;
	}
	/// consider another value
	void update(const Value& v)
	{
		if (cnt) {
			if (v > max) max = v;
			if (v < min) min = v;
			sum += v;
			sms += sqr(v);
		}
		else {
			sum = min = max = v;
			sms = sqr(v);
		}
		++cnt;
	}
	/// consider another value count times
	void update(const Value& v, int count)
	{
		if (cnt) {
			if (v > max) max = v;
			if (v < min) min = v;
			sum += count*v;
			sms += count*sqr(v);
		}
		else {
			sum = min = max = count*v;
			sms = count*sqr(v);
		}
		cnt += count;
	}
	/// get the average of the considered variables
	Value getAvg() const { return sum/cnt; }
	/// get the standard deviation of the considered variables
	Value getStd() const
	{
		Value mean = sum/cnt;
		Value E2 = sms;
		return sqrt (E2 / float (cnt) - mean*mean); 
	}
	/// get the sum of the considered variables
	Value getSum() const { return sum; }
	/// get the sum of the considered variables
	Value getSqrSum() const { return sms; }
	/// get the minimum of the considered variables
	Value getMin() const { return min; }
	/// get the maximum of the considered variables
	Value getMax() const { return max; }
	/// get the number of considered variables
	int   getCnt() const { return cnt; }
	int   getSms() const { return sms; }
	/// print out the statistics of the considered all values
	friend STD ostream& operator << (STD ostream& os, const Stats<Value>& s) {
		int w = os.width();
		int p = os.precision();
		os <<  "cnt=" << STD setw(w) << STD setprecision(p) << s.getCnt();
		os << " sum=" << STD setw(w) << STD setprecision(p) << s.getSum();
		os << " avg=" << STD setw(w) << STD setprecision(p) << s.getAvg();
		os << " std=" << STD setw(w) << STD setprecision(p) << s.getStd();
		os << " min=" << STD setw(w) << STD setprecision(p) << s.getMin();
		os << " max=" << STD setw(w) << STD setprecision(p) << s.getMax();
		os << " sms=" << STD setw(w) << STD setprecision(p) << s.getSms();
		return os;
	}
protected:
	Value min;
	Value max;
	Value sum;
	Value sms;
	int   cnt;
};

#endif

