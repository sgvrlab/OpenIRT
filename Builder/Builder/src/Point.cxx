#include "Point.h"
#include <float.h>
#include <limits.h>

//template struct Point<int>;
template struct Point<float>;
template struct Point<double>;

template<> void Point<int>::setMax() { setValue(INT_MAX); }
template<> void Point<int>::setMin() { setValue(INT_MIN); }
template<> void Point<float>::setMax() { setValue(FLT_MAX); }
template<> void Point<float>::setMin() { setValue(-FLT_MAX); }
template<> void Point<double>::setMax() { setValue(DBL_MAX); }
template<> void Point<double>::setMin() { setValue(-DBL_MAX); }
