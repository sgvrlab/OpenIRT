#include "stdafx.h"
#include "pair_mem.h"
#include <assert.h>
//#include "MyString.h" 	// for Basic Write function

using namespace std;

CPairMem::CPairMem (void)
{
	m_High = 0;
	m_Low = 0;
}
CPairMem::CPairMem (unsigned long High, unsigned long Low)
{
	m_High = High;
	m_Low = Low;
}
CPairMem CPairMem::operator + (unsigned long Add)
{
	CPairMem result = * this;

	unsigned long Remainder = result.m_Low + Add;
	if (Remainder <= result.m_Low)
		result.m_High++;
	result.m_Low = Remainder;

	return result;
 
}
CPairMem CPairMem::operator - (unsigned long Dec)
{
	CPairMem result = * this;

	unsigned long Remainder = result.m_Low - Dec;
	if (Remainder > result.m_Low)
		result.m_High--;
	result.m_Low = Remainder;

	return result;

}

CPairMem CPairMem::operator - (CPairMem & Dec)
{
	CPairMem result = * this;

	assert (result.m_High >= Dec.m_High);

	result.m_High -= Dec.m_High;

	result = result - Dec.m_Low;

	return result;

}


bool CPairMem::operator < (CPairMem & Dest)
{
	if (m_High < Dest.m_High)
		return true;
	else if (m_High == Dest.m_High)
		return (m_Low < Dest.m_Low);
	else
		return false;

	return true;
}

bool CPairMem::operator > (CPairMem & Dest)
{
	if (m_High > Dest.m_High)
		return true;
	else if (m_High == Dest.m_High)
		return (m_Low > Dest.m_Low);
	else
		return false;

	return true;
}

bool CPairMem::operator == (CPairMem & Dest)
{
	return (m_High == Dest.m_High && m_Low == Dest.m_Low);
}

bool CPairMem::operator >= (CPairMem & Dest)
{
	return (* this > Dest || * this == Dest);
}

bool CPairMem::operator <= (CPairMem & Dest)
{
	return (* this < Dest || * this == Dest);
}

ostream & Write (ostream & out, CPairMem & Pair)
{
	out << Pair.m_High;
	out << Pair.m_Low;

	return out;
}

istream & Read (istream & in, CPairMem & Pair)
{
	in >> Pair.m_High;
	in >> Pair.m_Low;

	return in;

}

