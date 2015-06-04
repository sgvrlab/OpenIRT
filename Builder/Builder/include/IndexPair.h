#ifndef INDEXHEADER_H
#define INDEXHEADER_H

// the fraction of inCoreStorage over the number of cashed boxes determines the bit split in the IndexPair class
#define INCORE_OVER_NRCASHED 1

#if INCORE_OVER_NRCASHED == 99 
#define MAX_NR_BI (32768*2)
#define MAX_LI (65535)
#define LI_BITS 16
#endif


#if INCORE_OVER_NRCASHED == 1 
#define MAX_NR_BI 32768
#define MAX_LI 131071
#define LI_BITS 17
#endif

#if INCORE_OVER_NRCASHED == 2 
#define MAX_NR_BI 16384
#define MAX_LI 262143
#define LI_BITS 18
#endif

#if INCORE_OVER_NRCASHED == 4 
#define MAX_NR_BI 8192
#define MAX_LI 524287
#define LI_BITS 19
#endif

#if INCORE_OVER_NRCASHED == 8
#define MAX_NR_BI 4096
#define MAX_LI 1048575
#define LI_BITS 20
#endif

#if INCORE_OVER_NRCASHED == 16
#define MAX_NR_BI 2048
#define MAX_LI 2097151
#define LI_BITS 21
#endif

#if INCORE_OVER_NRCASHED == 64
#define MAX_NR_BI 512
#define MAX_LI (2097151*2*2)
#define LI_BITS 23
#endif


// 33554431
struct IndexPair
{
public :
	///*
	// local index
	unsigned int li : LI_BITS;
	// box index
	unsigned int bi : 32-LI_BITS;
	//*/
	/*
	unsigned int li;
	// box index
	unsigned int bi;
	*/

	// default constructor
	IndexPair() : bi(0), li(0) {}
	// convert from int
	IndexPair(int i) { *((int*)this) = i; }
	// construct
	IndexPair(int _bi, int _li) : bi(_bi), li(_li) {}
	// return int rep
	int toInt() const { return *((int*)this); }
	// set the index
	void set(int _bi, int _li) { bi = _bi; li = _li; }

	// assign operator
	IndexPair & operator = (const IndexPair& I) { bi = I.bi; li = I.li; return *this;}

	// compare as single index
	bool operator < (const IndexPair& I) const { return (unsigned int&) (*this) < (unsigned int&) I; }
	// compare as single index
	bool operator > (const IndexPair& I) const { return (unsigned int&) (*this) > (unsigned int&) I; }
	// compare as single index
	bool operator == (const IndexPair& I) const { return (unsigned int&) (*this) == (unsigned int&) I; }
	// compare as single index
	bool operator != (const IndexPair& I) const { return (unsigned int&) (*this) != (unsigned int&) I; }
	// stream output
	friend std::ostream& operator << (std::ostream& os, const IndexPair& I) { 
		return os << I.bi << std::string(":") << I.li; }
};
#endif