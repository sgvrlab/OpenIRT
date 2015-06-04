#include "Defines.h"
#include <stdarg.h>

STD ostream* errorOut   = &(STD cerr);
STD ostream* warningOut = &(STD cerr);
STD ostream* debugOut   = &(STD cout);

///
string toString(unsigned int v)
{
	string res;
	char tmp[2];
	tmp[1]=0x0;
	if (v != 0) {
		while (v != 0) {
			tmp[0]='0'+(char)(v%10);
			res.insert(0, &tmp[0]);
			v /= 10;
		}
	}
	else res = "0";
	return res;
}

///
string toString(unsigned int v, int l, char fill)
{
	string res = toString(v);
	char tmp[2];
	tmp[1] = 0x0;
	tmp[0] = fill;
	while (res.length() < l) 
		res.insert(0,&tmp[0]);
	return res;
}

/// convert a time in seconds to a string of type hh:mm:ss
string splitTime(float t)
{
	int ti = t;
	int h = ti/3600;
	ti = ti % 3600;
	int m = ti/60;
	int s = ti%60;
	return toString(h,2,'0')+':'+toString(m,2,'0')+':'+toString(s,2,'0');
}

bool dprintf(const char* fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char buffer[1024];
	int nrChar = vsprintf(buffer, fmt, args);
	va_end(args);
	buffer[nrChar] = 0;
	PRINT(buffer);
	return true;
}

