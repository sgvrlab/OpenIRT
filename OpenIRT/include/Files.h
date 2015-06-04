#ifndef MYFILES_H
#define MYFILES_H

#include <stdio.h>
#include <string>
#include "Defines.h"

using namespace std;
/// return the filename of an indexed out of core file
//inline string getOutOfCoreName(const string& baseName, int bi) { return baseName+toString(bi,5,'0')+".ooc"; }
/// return the size of a file
extern int fileSize(const string& fileName);
/// return the size of a file
extern __int64 fileSize64(const string& fileName);
/// check whether a file exists
extern bool fileExists(const string& fileName);
/// open a gzipped file as if it was a regular file
extern FILE* fopenGzipped(const char* filename, const char* mode);
/// check if directory exists
bool dirExists(const string& dirName);
/// create a new directory
bool mkdir(const string& dirName);
/// remove a directory
bool rmdir(const string& dirName);


/// read a file into a newly allocated memory block
template <class T> T* readFile(const string& fileName, int& nrElements)
{
	__int64 nr = fileSize64(fileName);
	if (nr == -1) return 0;
	nr /= sizeof(T);

	T* elements = new T[nr];
	if (!readFileVoid(fileName, elements, nr, sizeof(T))) {
		delete elements;
		return 0;
	}
	nrElements = nr;
	return elements;
}

/// read a file into a newly allocated memory block
template <class T> T* readFile(const string& fileName, int& nrElements, int & MaxSize, char * pDummy)
{
	__int64 nr = fileSize64(fileName);
	if (nr == -1) return 0;
	nr /= sizeof(T);

	if (nr < 512)
		MaxSize = nr  + 50;
	else
		MaxSize = nr * 1.05;

	if (MaxSize - nr < 50)
		MaxSize = nr + 50;
	
	T* elements = new T[MaxSize];
	if (!readFileVoid(fileName, elements, nr, sizeof(T))) {
		delete elements;
		return 0;
	}
	nrElements = nr;
	return elements;
}


/// read a file into a newly allocated memory block
template <class T> T* readFile(const string& fileName, unsigned long offset, int& nrElements)
{
	T* elements = new T[nrElements];
	if (!readFileVoid(fileName, elements, offset, nrElements, sizeof(T))) {
		delete elements;
		return 0;
	}
	return elements;
}
/// read a file into a newly allocated memory block
template <class T> T* readFile64(const string& fileName, __int64 offset, int& nrElements)
{
	T* elements = new T[nrElements];
	if (!readFileVoid64(fileName, elements, offset, nrElements, sizeof(T))) {
		delete elements;
		return 0;
	}
	return elements;
}
/// write a file from a given memory block and element count and return whether this succeeded
template <class T> bool writeFile(const string& fileName, T* elements, int nrElements)
{
	return writeFileVoid(fileName,elements,nrElements,sizeof(T));
}
/// write a block of a file from a given memory block and element count to a given offset in the file and return whether this succeeded
template <class T> bool writeFile(const string& fileName, T* elements, int offset, int nrElements)
{
	return writeFileVoid(fileName,elements,offset,nrElements,sizeof(T));
}
/// write a block of a file from a given memory block and element count to a given offset in the file and return whether this succeeded
template <class T> bool writeFile64(const string& fileName, T* elements, __int64 offset, int nrElements)
{
	return writeFileVoid64(fileName,elements,offset,nrElements,sizeof(T));
}
/// append a given memory block of element count elements to a given file and return whether this succeeded
template <class T> bool appendFile(const string& fileName, T* elements, int nrElements)
{
	return appendFileVoid(fileName,elements,nrElements,sizeof(T));
}
/// delete a file completely
extern bool deleteFile(const string& fileName);
/// read a file into a given allocated memory block
extern bool readFileVoid(const string& fileName, void* dest, int nrElements, int elementSize);
/// read a block of a file into a given memory block
extern bool readFileVoid(const string& fileName, void* dest, unsigned long offset, int nrElements, int elementSize);
/// read a block of a file into a given memory block
extern bool readFileVoid64(const string& fileName, void* dest, __int64 offset, int nrElements, int elementSize);
/// write a file from a given memory block and element count and return whether this succeeded
extern bool writeFileVoid(const string& fileName, void* elements, int nrElements, int elementSize);
/// write a block of a file from a given memory block and element count and return whether this succeeded
extern bool writeFileVoid(const string& fileName, void* elements, int offset, int nrElements, int elementSize);
/// write a block of a file from a given memory block and element count and return whether this succeeded
extern bool writeFileVoid64(const string& fileName, void* elements, __int64 offset, int nrElements, int elementSize);
/// append a given memory block of element count elements to a given file and return whether this succeeded
extern bool appendFileVoid(const string& fileName, void* elements, int nrElements, int elementSize);



#endif