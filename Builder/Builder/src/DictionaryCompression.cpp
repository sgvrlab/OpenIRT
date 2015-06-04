#include "DictionaryCompression.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>

bool compareFreq(pair<unsigned int, unsigned int> &x, pair<unsigned int, unsigned int> &y)
{
	return x.second > y.second;
}

DictionaryCompression::DictionaryCompression(int nbits)
: nbits(nbits)
{
	corr_range = (unsigned int)(pow(2.0f, nbits));
	dicHashTable = new DicHashTable();
	dicTable = new int[4096];
	sizeDicTable = 0;
	corr_min = -(corr_range/2);
	corr_max = corr_min + corr_range - 1;
	bitsSizeHashTable = 1;
	hasDicTable = true;
	encoder = 0;
	isFirstDecode = true;
	last = 0;
	curInput = 0;
}
DictionaryCompression::DictionaryCompression(int nbits, int *dicTable)
: nbits(nbits), dicTable(dicTable)
{
	corr_range = (unsigned int)(pow(2.0f, nbits));
	dicHashTable = new DicHashTable();
	//dicTable = new int[4096];
	sizeDicTable = 0;
	corr_min = -(corr_range/2);
	corr_max = corr_min + corr_range - 1;
	bitsSizeHashTable = 1;
	hasDicTable = false;
	encoder = 0;
	isFirstDecode = true;
	last = 0;
	curInput = 0;
}

DictionaryCompression::~DictionaryCompression()
{
	delete dicHashTable;
	if(hasDicTable)
		delete[] dicTable;
}

void DictionaryCompression::encode(unsigned int value)
{
#if 0
	DicHashTableIt it = dicHashTable->find(value);
	if(it == dicHashTable->end())
	{
		encoder->encode(bitsSizeHashTable, 0);
		encoder->encode(nbits, value);
		dicHashTable->insert(std::pair<unsigned int, unsigned int>(value, dicHashTable->size()));
		bitsSizeHashTable = getBits(dicHashTable->size());
	}
	else
	{
		encoder->encode(bitsSizeHashTable, it->second+1);
	}
#else
	input.push_back(value);

	DicHashTableIt it = dicHashTable->find(value);
	if(it == dicHashTable->end())
	{
		dicHashTable->insert(std::pair<unsigned int, unsigned int>(value, 1));
	}
	else
	{
		it->second++;
	}
#endif
}

	void DictionaryCompression::encodeNone(int real)
	{
		int corr = real - last;
		if (corr < corr_min) corr += corr_range;
		else if (corr > corr_max) corr -= corr_range;
		encode(corr - corr_min);
		last = real;
	}

	void DictionaryCompression::encodeLast(int pred, int real)
	{
		int corr = real - pred;
		if (corr < corr_min) corr += corr_range;
		else if (corr > corr_max) corr -= corr_range;
		encode(corr - corr_min);
		last = real;
	}

	void DictionaryCompression::encodeAcross(int pred, int real)
	{
		int corr = real - pred;
		if (corr < corr_min) corr += corr_range;
		else if (corr > corr_max) corr -= corr_range;
		encode(corr - corr_min);
		last = real;
	}


int DictionaryCompression::PrepareDecoding()
{
	numInputs = decoder->decodeInt();
	int size = decoder->decode(nbits);
	bitsSizeHashTable = getBits(size);
	for(int i=0;i<size;i++)
	{
		dicTable[i] = decoder->decode(nbits);
	}
	input.clear();
	unsigned int value;
	for(unsigned int i=0;i<numInputs;i++)
	{
		if((value = decoder->decode(bitsSizeHashTable)) == 0)
		{
			value = decoder->decode(nbits);
		}
		else
		{
			value = dicTable[value-1];
		}
		input.push_back(value);
	}
	return 1;
}

void DictionaryCompression::done()
{
#if 1
	vector< pair<unsigned int, unsigned int> > *data = new vector< pair<unsigned int, unsigned int> >;
	for(DicHashTableIt it = dicHashTable->begin(); it != dicHashTable->end(); ++it)
	{
		data->push_back(*it);
	}

	// sort by frequencies
	std::sort(data->begin(), data->end(), compareFreq);

	// normalize
	size_t numData = input.size();
	vector< pair<unsigned int, double> > *dataN = new vector< pair<unsigned int, double> >;
	for(size_t i=0;i<data->size();i++)
	{
		(*dataN).push_back(pair<unsigned int, double>((*data)[i].first, (*data)[i].second/(double)numData));
	}

	// find s which minimize the encoded data
	size_t s = 0;
	double minBits = DBL_MAX;
	double p = 0;
	double pp = 0;

	for(size_t i=1;i<=dataN->size();i++)
	{
		p += (*dataN)[i-1].second;

		double logS = ceil(log((double)i+1)/log(2.0));
		double C = nbits;
		double bits = (double)s*nbits/dataN->size() + p*logS + (1-p)*(C+logS);
		if(bits < minBits)
		{
			minBits = bits;
			s = i;
			pp = p;
		}
	}

	/*
	// make hash for searching dictionary index
	dicHashTable->clear();
	for(size_t i=0;i<s;i++)
	{
		dicHashTable->insert(pair<unsigned int, unsigned int>(data[i].first, i));
	}

	// encode data
	DicHashTable curDicHashTable;
	vector<unsigned int> temp;
	for(unsigned int i=0;i<input.size();i++)
	{
		DicHashTableIt it = curDicHashTable.find(input[i]);
		if(it == curDicHashTable.end())
		{
			DicHashTableIt it2 = dicHashTable->find(input[i]);
			if(it2 == dicHashTable->end())
			{
				// the value is out of range
				encoder->encode(bitsSizeHashTable, 0);
				encoder->encode(nbits, input[i]);
			}
			else
			{
				// add dictionary table entry
				encoder->encode(bitsSizeHashTable, 1);
				encoder->encode(nbits, input[i]);
				curDicHashTable.insert(std::pair<unsigned int, unsigned int>(input[i], (unsigned int)curDicHashTable.size()));
				bitsSizeHashTable = getBits((unsigned int)curDicHashTable.size()+1);
				temp.push_back(input[i]);
			}
		}
		else
		{
			encoder->encode(bitsSizeHashTable, it->second+2);
		}
	}
	*/
	// make hash for searching dictionary index and encode
	dicHashTable->clear();
	encoder->encodeInt(input.size());
	encoder->encode(nbits, s);
	for(size_t i=0;i<s;i++)
	{
		dicHashTable->insert(pair<unsigned int, unsigned int>((*data)[i].first, i));
		encoder->encode(nbits, (*data)[i].first);
	}
	bitsSizeHashTable = getBits((unsigned int)dicHashTable->size());

	// encode data
	for(unsigned int i=0;i<input.size();i++)
	{
		DicHashTableIt it = dicHashTable->find(input[i]);
		if(it == dicHashTable->end())
		{
			// the value is out of range
			encoder->encode(bitsSizeHashTable, 0);
			encoder->encode(nbits, input[i]);
		}
		else
		{
			encoder->encode(bitsSizeHashTable, it->second+1);
		}
	}
	delete data;
	delete dataN;
#endif
}

int DictionaryCompression::decode() 
{
#if 0
	int value;
	if(value = decoder->decode(bitsSizeHashTable))
	{
		value = dicTable[value-1];
	}
	else
	{
		value = decoder->decode(nbits);
		dicTable[sizeDicTable++] = value;
		bitsSizeHashTable = getBits(sizeDicTable);
	}
	return value;
#else
	unsigned int value = input[curInput++];
	/*
	if((value = decoder->decode(bitsSizeHashTable)) > 1)
	{
		value = dicTable[value-2];
	}
	else if(value == 1)
	{
		value = decoder->decode(nbits);
		dicTable[sizeDicTable++] = value;
		bitsSizeHashTable = getBits(sizeDicTable+1);
	}
	else
	{
		value = decoder->decode(nbits);
	}
	*/
#endif
	return value;
}

	int DictionaryCompression::decodeNone()
	{
		int real = decode() + corr_min + last;
		if (real < 0) real += corr_range;
		else if (real >= corr_range) real -= corr_range;
		last = real;
		return real;
	}

	int DictionaryCompression::decodeLast(int pred)
	{
		int real = decode() + corr_min + pred;
		if (real < 0) real += corr_range;
		else if (real >= corr_range) real -= corr_range;
		last = real;
		return real;
	}

	int DictionaryCompression::decodeAcross(int pred)
	{
		int real = decode() + corr_min + pred;
		if (real < 0) real += corr_range;
		else if (real >= corr_range) real -= corr_range;
		last = real;
		return real;
	}


/*
void DictionaryCompression::encodeU(unsigned int value) 
{
//if(value > corr_range) 
//	cout << "!!!!! out of range!" << endl;
DicHashTableIt it = dicHashTable->find(value);
if(it == dicHashTable->end())
{
encoder->encode(bitsSizeHashTable, 0);
encoder->encode(nbits, value);
dicHashTable->insert(std::pair<unsigned int, unsigned int>(value, dicHashTable->size()));
bitsSizeHashTable = getBits(dicHashTable->size());
}
else
{
encoder->encode(bitsSizeHashTable, it->second+1);
}
}

void DictionaryCompression::encode(int corr)
{
if (corr < corr_min) corr += corr_range;
else if (corr > corr_max) corr -= corr_range;
encodeU(corr - corr_min);
}

int DictionaryCompression::decodeU() 
{
int value;
if(value = decoder->decode(bitsSizeHashTable))
{
value = dicTable[value-1];
}
else
{
value = decoder->decode(nbits);
dicTable[sizeDicTable++] = value;
bitsSizeHashTable = getBits(sizeDicTable);
}
return value;
}

int DictionaryCompression::decode()
{
int corr = decodeU() + corr_min;
if (corr < 0) corr += corr_range;
else if (corr >= corr_range) corr -= corr_range;
return corr;
}
*/

/*
void DictionaryCompression::done()
{
if(encoder)
encoder->done();
}
*/
