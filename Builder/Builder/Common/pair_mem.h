#ifndef _Layout_pair_mem_
#define _Layout_pair_mem_

#include <fstream>

class CPairMem {
public :
	unsigned long m_High, m_Low;

	CPairMem (void);
	CPairMem (unsigned long High, unsigned long Low);

	CPairMem operator + (unsigned long Add);
	CPairMem operator - (unsigned long Dec);
	CPairMem operator - (CPairMem & Dec);
	bool operator < (CPairMem & Dest);
	bool operator > (CPairMem & Dest);
	bool operator == (CPairMem & Dest);
	bool operator >= (CPairMem & Dest);
	bool operator <= (CPairMem & Dest);

	friend std::ostream & Write (std::ostream & out, CPairMem & Pair);
	friend std::istream & Read (std::istream & in, CPairMem & Pair);
};


#endif

