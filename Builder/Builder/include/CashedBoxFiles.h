#ifndef CASHEDBOXFILES_H
#define CASHEDBOXFILES_H

#include "Files.h"
#include "IndexPair.h"
#include <assert.h>

#pragma warning( disable : 4018 )

struct AbstCashEntry
{
	AbstCashEntry() 
	{
		// Initially, maxsize = 0
		// need to manually turn on for preparing room.
		MaxSize = 0;	
		
		m_IsDataReady = 0;
			 
		size = 0;
		modified = 0;
		bi = -1;
		hits = 0;

	}
	/// 
	bool empty() const { return size == 0; }
	///
	virtual void dismiss() { size = 0; modified = 0; bi = -1; hits = 0; }

	// for foreign triangles during boundary simplification
	// we keep some room for growing data.
	int MaxSize;

	// to see if the dynamic data is set.
	int m_IsDataReady;



	/// nr of elements
	int size;
	/// whether this entry was modified
	char modified;
	/// the index of the cashed box
	short bi;
	/// index of the next cash
	int nextCash;
	/// index of the prev cash
	int prevCash;
	/// number of cash hits
	int hits;
};

/** one entry of the cash */
template <class T>
struct CashEntry : public AbstCashEntry
{
	/// pointer to the elements
	T* elements;
	/// return the factor between elements and stored units
	static int sizeScaleFactor() { return 1; }
	/// standard constructor, set empty entry
	CashEntry() : elements(0) {}
	/// destructor
	~CashEntry() { if (elements) delete [] elements; }
	/// dismiss a file
	void dismiss() { delete [] elements; elements = 0; AbstCashEntry::dismiss(); }
	/// write to file
	bool write(const string& fileName) const { return writeFile<T>(fileName, elements, size); }
	/// write block of file
	bool write(const string& fileName, int offset) const { return writeFile<T>(fileName, elements, offset, size); }
	/// write block of file
	bool write64(const string& fileName, __int64 offset) const { return writeFile64<T>(fileName, elements, offset, size); }
	/// read from file
	bool read(const string& fileName) 
	{ 
		if (elements) delete [] elements;
		

		if (MaxSize != 0)
			elements = readFile<T>(fileName,size, MaxSize, (char* )NULL);
		else
			elements = readFile<T>(fileName,size);

		return elements != 0;
	}
	/// read from file
	bool read(const string& fileName, unsigned long offset, int count) 
	{ 
		if (elements) delete [] elements;
		elements = readFile<T>(fileName, offset, count);
		size     = count;
		return elements != 0;
	}
	/// read from file
	bool read64(const string& fileName, __int64 offset, int count) 
	{ 
		if (elements) delete [] elements;
		elements = readFile64<T>(fileName, offset, count);
		size     = count;
		return elements != 0;
	}
	/// create file
	static void create(const string& fileName, int size)
	{
		T* ptr = new T[size];
		writeFile<T>(fileName,ptr,size);
		delete [] ptr;
	}
	/// read access to element
	const T& getElement(int i) const { return elements[i]; }
	/// read access to element
	T& refElement(int i) const { return elements[i]; }
	/// write access to element
	void setElement(int i, const T& v) { elements[i] = v; }
};

#if 0
/** one entry of the cash */
template struct CashEntry<bool> : public AbstCashEntry
{
	/// store one value
	bool tmp;
	/// pointer to the elements
	unsigned char* elements;
	/// return the factor between elements and stored units
	static int sizeScaleFactor() { return 8; }
	/// standard constructor, set empty entry
	CashEntry() : elements(0) {}
	/// destructor
	~CashEntry() { if (elements) delete [] elements; }
	/// compute the size in bytes
	static int byteSize(int s) { return (s+7)>>3; }
	/// dismiss a file
	void dismiss() { delete [] elements; elements = 0; AbstCashEntry::dismiss(); }
	/// write to file
	bool write(const string& fileName) const { return writeFile<unsigned char>(fileName, elements, byteSize(size)); }
	/// write to file
	bool write(const string& fileName, int offset) const { return writeFile<unsigned char>(fileName, elements, byteSize(offset), byteSize(size)); }
	/// read from file
	bool read(const string& fileName) 
	{ 
		if (elements) delete [] elements;
		elements = readFile<unsigned char>(fileName,size);
		size *= 8;
		return elements != 0;
	}
	/// read from file
	bool read(const string& fileName, int offset, int count) 
	{ 
		if (elements) delete [] elements;
		elements = readFile<unsigned char>(fileName,offset,count);
		size = 8*count;
		return elements != 0;
	}
	/// create file
	static void create(const string& fileName, int size)
	{
		int bs = byteSize(size);
		unsigned char* ptr = new unsigned char[bs];
		STD fill(ptr, ptr+bs, (unsigned char) 0);
		writeFile<unsigned char>(fileName,ptr,bs);
		delete [] ptr;
	}
	/// read access to element
	const bool& getElement(int i)
	{
		int li = i&7;
		int bi = i>>3;
		tmp = ( elements[bi] & pow2(li) ) != 0;
		return tmp;
	}
	/// write access to element
	bool& refElement(int i)
	{
		int li = i&7;
		int bi = i>>3;
		tmp = ( elements[bi] & pow2(li) ) != 0;
		return tmp;
	}
	/// write access to element
	void setElement(int i, bool v) 
	{ 
		int li = i&7;
		int bi = i>>3;
		if (v) elements[bi] |= pow2(li);
		else elements[bi] &= ~((unsigned char) pow2(li));
	}
};
#endif

class AbstCashedBoxFiles
{
public:	
	// If this flag is 0, we don't need to write
	int m_Write;



	// if this is on, we prepare extran room for grong of content of cache.
	int m_NeedExtraRoom;


	/// store the maximum number of cashed boxes
	int nrCashed;
	/// whether to print debug info
	bool debug;
	/// store the base file name
	string baseFileName;
	/// store the number of boxes
	int nrBoxes;
	/// for each box store a short index to specify the cash index or -1 if not cashed
	short* cashIndex;
	/// store the index of the first cash entry
	int firstCashEntry;
	/// store the index of the last cash entry
	int lastCashEntry;
	/// keep track of the total amount of elements read from disk
	int readElements;
	/// keep track of the total amount of elements written to disk
	int writtenElements;
	/// the number of read boxes
	int readBoxes;
	/// the number of written boxes
	int writtenBoxes;
	/// return the file name of a box
	string getFileName(int bi) const { return getOutOfCoreName(baseFileName,bi); }
	/// write a cash box of given cash index only if modified
	virtual bool writeCashBox(int ci) = 0;
	/// read a cash box from a given file
	virtual bool readCashBox(int ci, int bi) = 0;

	virtual bool InitCashBox (int ci, int bi) = 0;

	/// get reference to cash
	virtual AbstCashEntry& refCashEntry(int ci) = 0;
	/// ensure that a given box is cashed and return its cash index
	short ensureCashed(int bi);
	short ensureCashedWithoutRead (int bi);
public:
	/// construct
	AbstCashedBoxFiles(const string& baseName, int _nrBoxes, int _nrCashed, bool _debug = false);
	/// destructor
	virtual ~AbstCashedBoxFiles();
	/// set counters to 0
	void initCounters() { readElements = writtenElements = readBoxes = writtenBoxes = 0; }
	/// write all modified blocks to disk
	void flush();
	/// return the number of read elements
	int getNrReadElements() const { return readElements; }
	/// return the number of written elements
	int getNrWrittenElements() const { return writtenElements; }
	/// return the number of cashed boxes
	int getNrCashed() const { return nrCashed; }
	/// return the ci-th cashed entry
	int getCashedBox(int ci);
	/// return a short version of filename
	string getShortFileName() const;
	/// return a short version of filename
	string getShortFileName(int bi) const;
	/// remove all files
	void clear() { for (int bi = 0; bi < nrBoxes; ++bi) deleteFile(getFileName(bi)); }
	/// empty the cash
	void makeEmpty();
	/// show statistics
	virtual void showStats() = 0;
};

/** uses an out of core field of elements of type T and provides random access with a given number
    of cashed boxes. */
template <class T>
class CashedBoxFiles : public AbstCashedBoxFiles
{
public:
	
	/// the cash entries
	CashEntry<T>* cash;
	/// write a cash box of given cash index only if modified
	virtual bool writeCashBox(int ci);
	/// read a cash box from a given file
	virtual bool readCashBox(int ci, int bi);

	virtual bool InitCashBox (int ci, int bi);
	
	/// get reference to cash
	virtual AbstCashEntry& refCashEntry(int ci) { return cash[ci]; }
public:
	/// construct the file from the total number of boxes and the maximum number of cashed boxes
	CashedBoxFiles(const string& baseName, int _nrBoxes, int _nrCashed, bool _debug = false, int NeedExtraRoom = 0);
	/// destructor
	~CashedBoxFiles();
	/// constant access to elements
	const T& getElement(IndexPair I) {
		static T tmp;
		IFWARN(I.bi >= nrBoxes, 0, "box index " << I.bi << " out of range [0," << nrBoxes << ")") {
			//printf ("hello\n");
			exit(-1);
			return tmp;
		}
		short ci = ensureCashed(I.bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") {
			//printf ("ehllo\n");
			return tmp;
		}
		IFWARN(I.li >= cash[ci].size, 0, "element index " << I.li << " out of range [0," << cash[ci].size << ")") {
			//printf ("hello\n");
			return tmp;
		}
		return cash[ci].getElement(I.li);
	} 
	const T * getElements(int bi) {
		static T tmp;
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return NULL;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return NULL;
		return (const T *) cash[ci].elements;
	}	

	int getNumElements(int bi) {
		static T tmp;
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return NULL;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return NULL;
		return cash[ci].size;
	}	

	int & refDataReady (int bi) {
		static int tmp;
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return tmp;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return tmp;
		return  cash[ci].m_IsDataReady;

	}

	void setUnReady (int bi)
	{
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return;
		short ci = ensureCashed(bi);
		cash [ci].m_IsDataReady = 0;
	}

	void freeElements (int bi) {
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return;

		short ci = ensureCashed(bi);
 
		cash[ci].modified = 0;
		//delete [] cash[ci].elements;
		//cash[ci].elements = NULL;
		cash[ci].size = 0;

		// BUG: this can be a bug
		// we have to set this max size dynamically.
		//cash[ci].MaxSize = 0;
	}

	void removeFile (int bi) {
		deleteFile(getFileName(bi));

	}

	int & refNumElements(int bi) {
		static int tmp;
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return tmp;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") 
			return tmp;
		return  cash[ci].size;
	}	

	T& getConstElement(IndexPair I) {
		static T tmp;
		IFWARN(I.bi >= nrBoxes, 0, "box index " << I.bi << " out of range [0," << nrBoxes << ")") 
			return tmp;
		short ci = ensureCashed(I.bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return tmp;
		IFWARN(I.li >= cash[ci].size, 0, "element index " << I.li << " out of range [0," << cash[ci].size << ")") 
			return tmp;
		return cash[ci].refElement(I.li);
	}

	T& HasOneRoom (int bi) {
		static T tmp;
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return tmp;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return tmp;
		cash[ci].modified = 1;

		if (cash [ci].MaxSize < cash [ci].size + 1) {
			printf ("Need to increase room in the content\n");
			exit (-1);
		}

		return cash [ci].elements [cash [ci].size++];
	}
 
	/// write access to elements, sets the modified flag
	T& refElement(IndexPair I) {
		static T tmp;
		IFWARN(I.bi >= nrBoxes, 0, "box index " << I.bi << " out of range [0," << nrBoxes << ")") {
			exit(-1);
			return tmp;
		}
		short ci = ensureCashed(I.bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return tmp;
		cash[ci].modified = 1;
		IFWARN(I.li >= cash[ci].size, 0, "element index " << I.li << " out of range [0," << cash[ci].size << ")")  {
			//printf ("hello\n");
			return tmp;
		}
		return cash[ci].refElement(I.li);
	}

	/// write access to elements, sets the modified flag
	T * refElements (int bi) {
		static T tmp;
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") 
			return NULL;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return NULL;
		cash[ci].modified = 1;
		return (T *)cash[ci].elements;
	}



	/// write access to elements, sets the modified flag
	void setElement(IndexPair I, const T& value) {
		IFWARN(I.bi >= nrBoxes, 0, "box index " << I.bi << " out of range [0," << nrBoxes << ")") return;
		short ci = ensureCashed(I.bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return;
		cash[ci].modified = 1;
		IFWARN(I.li >= cash[ci].size, 0, "element index " << I.li << " out of range [0," << cash[ci].size << ")") return;
		cash[ci].setElement(I.li,value);
	}
	void setElements (int bi, T* values, int size) {
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") return;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return;
		cash[ci].modified = 1;
		cash[ci].size = size;
		cash[ci].m_IsDataReady = 0;

		if (cash[ci].elements != NULL)
			delete [] cash[ci].elements;

		cash[ci].elements = values;
	}
	void setElements (int bi, T* values, int size, int MaxSize) {
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") return;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return;
		cash[ci].modified = 1;
		cash[ci].size = size;
		cash[ci].MaxSize = MaxSize;
		cash[ci].m_IsDataReady = 0;

		if (cash[ci].elements != NULL)
			delete [] cash[ci].elements;

		cash[ci].elements = values;
	}

	void initElements (int bi, T* values, int size) {
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") return;
		short ci = ensureCashedWithoutRead (bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return;
		cash[ci].modified = 1;
		cash[ci].size = size;

		/*
		if (cash[ci].elements != NULL)
			delete [] cash[ci].elements;
		*/
		cash[ci].elements = values;
	}
 


	void Fetch (int bi) {
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return;
	}

	


	void setNumElements (int bi, int size) {
		IFWARN(bi >= nrBoxes, 0, "box index " << bi << " out of range [0," << nrBoxes << ")") return;
		short ci = ensureCashed(bi);
		IFWARN(ci == -1, 0, "could not cash a box!!!") return;
		cash[ci].modified = 1;
		cash[ci].size = size;
	}

	// to see if data is in the core
	bool HasElementsInCore (int bi) {
		short ci = cashIndex[bi];
		// if box is not already cashed
		if (ci == -1) 
			return false;
 
		short new_ci = ensureCashed(bi);
		assert (ci == new_ci);

		return true;
	}


	/// fill new files with empty entries
	void attach(const string& parentFileName, int size);
	/// instead of attaching, one can create each file by hand
	void create(int bi, int size) { CashEntry<T>::create(getFileName(bi), size); }
	/// show statistics
	void showStats();
};


/** uses an out of core field of elements of type T and provides random access with a given number
    of cashed boxes. */
template <class T>
class CashedBoxSingleFile : public CashedBoxFiles<T>
{
protected:
	/// store the size of a box for the case of a single file
	int boxSize;
	/// write a cash box of given cash index only if modified
	virtual bool writeCashBox(int ci);
	/// read a cash box from a given file
	virtual bool readCashBox(int ci, int bi);
	/// return the size of the last box
	int lastBoxSize() const { return nrElements - (nrBoxes-1)*boxSize; }
public:
	/// store the total number of elements in the file for single file mode
	int nrElements;
	/// construct the file from the total number of boxes and the maximum number of cashed boxes
	CashedBoxSingleFile(const string& baseName, int _nrBoxes, int _nrCashed, bool _debug);
	/// show statistics
	void showStats();
	/// access an element with a global index
	T operator [] (int i) { return getElement(IndexPair(i/boxSize,i%boxSize)); } 
	T& getRef (int i) { return refElement(IndexPair(i/boxSize,i%boxSize)); }
};

#pragma warning( default : 4018 )

#endif
