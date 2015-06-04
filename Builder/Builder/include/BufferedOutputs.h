#ifndef BUFFERED_OUTPUTS_H
#define BUFFERED_OUTPUTS_H

#include "Defines.h"
#include "Files.h"
#include <vector>

/** use a collection of files for output */
template <class T>
class BufferedOutputs
{
protected:
	/// vector of output buffers
	STD vector<T*> buffers;
	/// position in each buffer
	STD vector<int> positions;
	/// position in each buffer
	STD vector<int> sizes;
	/// store the buffer size
	int bufferSize;
	/// whether we hace a single file
	bool single;
	/// base file name
	string fileName;
	/// start offset of box index
	int biOffset;
public:
	/// construct for given number of buffers and buffer size
	BufferedOutputs(const string& _fileName, int nrBuffers, int _bufferSize, int _biOffset = 0) : fileName(_fileName), bufferSize(_bufferSize), biOffset(_biOffset)
	{
		//printf ("buffer size is %d\n", _bufferSize);
		buffers.resize(nrBuffers);
		positions.resize(nrBuffers);
		sizes.resize(nrBuffers);
		for (int bi=0; bi<nrBuffers; ++bi) {
			buffers[bi] = new T[bufferSize];
			positions[bi] = 0;
			sizes[bi] = 0;
		}
		single = false;
	}
	/// construct for only one buffered stream if given size
	BufferedOutputs(string _fileName, int _bufferSize) : fileName(_fileName), bufferSize(_bufferSize), biOffset(0)
	{
		//printf ("buffer size is %d\n", _bufferSize);
		buffers.push_back(new T[bufferSize]);
		positions.push_back(0);
		sizes.push_back(0);
		single = true;
	}
	/// destructor flushes before destruction
	~BufferedOutputs()
	{
		flush();
		for (unsigned int bi=0; bi<buffers.size(); ++bi) delete [] buffers[bi];
	}
	/// return the file name of a given box
	string getFileName(int bi) const
	{
		return single ? fileName : getOutOfCoreName(fileName,bi+biOffset);
	}
	/// remove all existing files
	void clear()
	{
		for (unsigned int bi=0; bi<buffers.size(); ++bi) {
			string name = getFileName(bi); 
			if (fileExists(name)) deleteFile(name);
		}
	}
	/// write all remaining data to the files
	void flush()
	{
		for (unsigned int bi=0; bi<buffers.size(); ++bi)  
			if (positions[bi] > 0) {
				//if (bi%100 == 0) 
				//	printf ("bi = %d\n", bi);

				writeBuffer(bi, positions[bi]);
				positions[bi] = 0;
			}
	}
	/// write a buffer to its file
	bool writeBuffer(int bi, int size = -1) const
	{
		if (size == -1) 
			size = bufferSize;

		string name = getFileName(bi);
		FILE* fp = fopen(name.c_str(),"ab");
		if (!fp) 
			return false;
		if (size != fwrite(buffers[bi],sizeof(T),size,fp)) 
			return false;

		if (ferror (fp) != 0) {
			printf ("there is something wrong in writing in bufferoutput (bi = %d, size = %d)\n", bi, size);
			exit (-1);
		}
		return fclose(fp) == 0;
	}
	/// only reserve an element
	int reserveElement(int bi) { return sizes[bi]++; }
	/// only reserve an element
	int reserveElement() { return sizes[0]++; }
	/// occupy a previously reserved element
	void occupyElement(int bi, const T& v) 
	{ 
		int& pos = positions[bi];
		buffers[bi][pos] = v;
		if (++pos == bufferSize) {
			if (writeBuffer(bi) == false) {
				printf ("Something happened in the writing\n");
				exit (-1);
			}
			pos = 0;
		}
	}
	/// occupy a previously reserved element
	void occupyElement(const T& v) { occupyElement(0); } 
	/// only reserve an element
	int reserveElement(const T& v) { return sizes[0]++; }
	/// append an element to the i-th stream and return its new index
	int appendElement(int bi, const T& v)
	{
		occupyElement(bi-biOffset, v);
		return reserveElement(bi-biOffset);
	}
	/// append an element to first stream and return its new index
	int appendElement(const T& v) { return appendElement(0,v); }
};

#endif