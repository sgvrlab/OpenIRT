#ifndef DEFINES_H
#define DEFINES_H

#include <math.h>
#include <string>
#include <iostream>

/// type of byte
typedef unsigned char Byte;
/// calculate k^2 for k = 0..31
inline unsigned int pow2(int k)
{
	static const unsigned int powConst[32] = 
	{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648U };
	return powConst[k];
}

/// the number pi
//#define PI 3.14159265358979323846




/// square of a number
template <class T> T sqr(T t) { return t*t; }
/// round a floating point number
template <class Value> Value round(Value v) { return floor(v+0.5f); }
#ifndef max
#define max(a, b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#endif
/// macro for std
#define STD std::
/// make string available
using std::string;
/// convert to string
extern string toString(unsigned int v, int l, char fill = ' ');
/// convert to string
extern string toString(unsigned int v);
/// convert a time in seconds to a string of type hh:mm:ss
extern string splitTime(float t);

extern STD ostream* errorOut;
extern STD ostream* warningOut;
extern STD ostream* debugOut;
/// error output stream
#define EOUT (*errorOut)
/// warning output stream
#define WOUT (*warningOut)
/// debug output stream
#define DOUT (*debugOut)
/// link error output stream to given stream
inline void setEOUT(STD ostream& os)   { errorOut = &os; }
/// link warning output stream to given stream
inline void setWOUT(STD ostream& os)   { warningOut = &os; }
/// link debug output stream to given stream
inline void setDOUT(STD ostream& os)   { debugOut = &os; }
/// link all three output streams (error, warning and debug) to the same stream
inline void setALLOUT(STD ostream& os) { errorOut = warningOut = debugOut = &os; }
/// c-style output to debug stream
extern bool dprintf(const char* fmt, ...);
/// print a warning to the warning stream
/** output warning with number Nr and message Msg if condition is fulfilled. Message 
	is something you can write after a stream. The statement works in the same way as 
	an if-clause. A typical example could be:\\
	{\tt IFWARN(readError, 15, "couldn't read file " << filename) return false; }.\\
	CAREFULL: this statement is not save in a parallel program. */
#define IFWARN(Cond,Nr,Msg) if (Cond) WOUT << __FILE__ << '(' << __LINE__ << ") : warning " << Nr << ": " << Msg << STD endl; if (Cond) 
/// print a warning to the warning stream
#define WARN(Nr, Msg) WOUT << __FILE__ << '(' << __LINE__ << ") : warning " << Nr << ": " << Msg << STD endl;
/// print something to the debug stream
#define PRINT(Msg) DOUT << Msg;
/// print something to the debug stream followed by an endl
#define PRINTLN(Msg) DOUT << Msg << STD endl;
/// check whether an index is marked
template <class Index> inline bool  isMarked(Index i)    { return i<-1; }
/// return invalid index. The function argument is a dummy and only used to determine the template argument.
template <class Index> inline Index noIndex(Index)       { return -1; }
/// mark an index
template <class Index> inline Index markIndex(Index i)   { return (i<0) ? i : -i-2; }
/// unmark an index
template <class Index> inline Index unmarkIndex(Index i) { return (i<-1) ? -i-2 : i; }


#endif
