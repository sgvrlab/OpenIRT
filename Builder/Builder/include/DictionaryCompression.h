#ifndef DICTIONARY_COMPRESSION_HPP
#define DICTIONARY_COMPRESSION_HPP
#include <algorithm>
#include <iostream>
using namespace std;
#include "BitCompression.hpp"
#include <math.h>
#ifdef _WIN32
	#include <hash_map>
	using namespace std;
  #ifdef STDEXT
    using namespace stdext;
  #endif
#else
	#include <hash_map.h>
#endif

class DictionaryCompression
{
protected :
	int sizeDicTable;
	int nbits;
	int corr_range;
	int corr_min;
	int corr_max;
	int last;

	typedef stdext::hash_map<unsigned int, unsigned int> DicHashTable;
	typedef DicHashTable::iterator DicHashTableIt;
	DicHashTable *dicHashTable;

	vector<unsigned int> input;
	unsigned int curInput;

	unsigned int numInputs;
	int *dicTable;
	int hasDicTable;
	int bitsSizeHashTable;

	int isFirstDecode;

public :
	BitCompression *encoder;
	BitCompression *decoder;

	DictionaryCompression(int nbits);
	DictionaryCompression(int nbits, int *dicTable);
	~DictionaryCompression();

	void setEncoder(BitCompression *encoder) {this->encoder = encoder;}
	void setDecoder(BitCompression *decoder) {this->decoder = decoder;}

	void encode(unsigned int value);
	void encodeNone(int real);
	void encodeLast(int pred, int real);
	void encodeAcross(int pred, int real);


	int PrepareDecoding();
	void done();

	int decode();
	int decodeNone();
	int decodeLast(int pred);
	int decodeAcross(int pred);

	inline int getBits(unsigned int x)
	{
		if(x < 0)
			cout << "out of range : (getBits)" << endl;
		int bits = 1;
		while((x >>= 1) != 0) bits++;
		return bits;
	}
};

#endif