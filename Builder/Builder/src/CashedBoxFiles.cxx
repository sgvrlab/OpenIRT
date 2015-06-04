#include "CashedBoxFiles.h"

/*
#ifndef DAVIS
#include "Point.h"
#else
#include <Geometry/Point.h>
#endif
#include "TwinEdge.h"

#include "Cluster.h"


template class CashedBoxFiles<bool>;
template class CashedBoxFiles<unsigned short>;
#ifndef DAVIS
template class CashedBoxFiles<Point<int> >;
template class CashedBoxFiles<Point<float> >;
#else
template class CashedBoxFiles<Point<3,int> >;
template class CashedBoxFiles<Point<3,float> >;
#endif
template class CashedBoxFiles<IndexPair>;
template class CashedBoxFiles<TwinEdge<true> >;
template class CashedBoxFiles<TwinEdge<false> >;
template class CashedBoxSingleFile<IndexPair>;

// sungeui start -------------------------
typedef Point<float> InpPoint;
template class CashedBoxFiles<COutTriangle>;
template class CashedBoxFiles<CGeomVertex>;
template class CashedBoxFiles<MINQUADRIC4u>;
template class CashedBoxSingleFile<InpPoint>;
// sungeui end ----------------------------
*/

// christian
#include "Vector3.h"
#include "Vertex.h"
#include "Triangle.h"
#include "rgb.h"
#include "Point.h"
template class CashedBoxSingleFile<unsigned int>;
template class CashedBoxSingleFile<Vector3>;
template class CashedBoxSingleFile<Vertex>;
template class CashedBoxSingleFile<Triangle>;
template class CashedBoxSingleFile<rgb>;
template class CashedBoxSingleFile<rgba>;
template class CashedBoxSingleFile<Point<int> >;

/// write a cash box of given cash index only if modified
template <class T>
bool CashedBoxFiles<T>::writeCashBox(int ci)
{
	// check if box was modified
	CashEntry<T>& cashEntry = cash[ci];
	if (cashEntry.modified == 1) {

		if (m_Write == 1) {
			IFWARN(!cashEntry.write(getFileName(cashEntry.bi)), 0, "could not write box file " << getFileName(cashEntry.bi)) 
				exit(0);
			if (debug) PRINTLN("wrote " << ci << " file " << getShortFileName(cashEntry.bi));
			// accumulate number of written elements
			writtenElements += cashEntry.size;
			writtenBoxes += 1;
		}
		else {
			// Later, we neet to change it to call back function
			//SaveErrors (cashEntry.bi, cashEntry.size, (MINQUADRIC4u *) cashEntry.elements);

			/*
			delete [] cashEntry.elements;
			cashEntry.elements = NULL;
			*/

		} 

		cashEntry.modified = 0;
	}
	return true;
}

/// read a cash box from a given file
template <class T>
bool CashedBoxFiles<T>::readCashBox(int ci, int bi)
{
	// check if box was modified
	CashEntry<T>& cashEntry = cash[ci];

	if (m_Write == 1) {
		IFWARN(!cashEntry.read(getFileName(bi)), 0, "could not read box file " << getFileName(bi)) 
			return false;
		if (debug) PRINTLN("read in " << ci << " file  " << getShortFileName(bi));
		readElements += cashEntry.size;
		++readBoxes;
		cashEntry.m_IsDataReady = 0;
	}
	else {

		if (cashEntry.elements != NULL)
			delete [] cashEntry.elements;
		
		/*
		if (cashEntry.elements != NULL) {
			printf ("someting is wrong in load errors\n");
		}
		*/

		//LoadErrors (bi, cashEntry.size, (MINQUADRIC4u **) &cashEntry.elements);
		readElements += cashEntry.size;
		++readBoxes;
 

		cashEntry.m_IsDataReady = 0;
	}
 
	// update cash entry and cash index field
	cashEntry.modified = 0;
	cashEntry.hits     = 0;
	cashEntry.bi       = bi;
	cashIndex[bi]      = ci;
	return true;
}

template <class T>
bool CashedBoxFiles<T>::InitCashBox(int ci, int bi)
{
	// check if box was modified
	CashEntry<T>& cashEntry = cash[ci];


	// update cash entry and cash index field
	cashEntry.m_IsDataReady = 0;
	cashEntry.modified = 0;
	cashEntry.hits     = 0;
	cashEntry.bi       = bi;
	cashIndex[bi]      = ci;
	return true;
}

/// ensure that a given box is cashed
short AbstCashedBoxFiles::ensureCashed(int bi)
{
	short ci = cashIndex[bi];
	// if box is not already cashed
	if (ci == -1) {
		// check if the next available cash entry was already used
		ci = lastCashEntry;
		AbstCashEntry& cashEntry = refCashEntry(ci);
		if (!cashEntry.empty()) {
			// and write if necessary
			if (debug && (cashEntry.modified == 0)) {
				PRINT("dismissed " << getShortFileName(cashEntry.bi) << " cash: ");
				int cj = firstCashEntry;
				for (int i=0; i<nrCashed; ++i, cj = refCashEntry(cj).nextCash) PRINT(refCashEntry(cj).bi << ":" << refCashEntry(cj).hits << " ");
				PRINTLN(refCashEntry(lastCashEntry).bi);
			}
			writeCashBox(lastCashEntry);
			cashIndex[cashEntry.bi] = -1;
		}
		// read the new box
		if (!readCashBox(ci,bi)) return -1;
	}
	// move entry to front 
	if (ci != firstCashEntry) {
		// increment hits
		AbstCashEntry& cashEntry = refCashEntry(ci);
		++cashEntry.hits;
		// remove entry from list
		int pci = cashEntry.prevCash;
		IFWARN(pci == -1, 0, "no prev cash!!!") return ci;
		if (lastCashEntry == ci) lastCashEntry = pci;
		int nci = cashEntry.nextCash;
		refCashEntry(pci).nextCash = nci;
		if (nci != -1) refCashEntry(nci).prevCash = pci;
		// insert at front
		cashEntry.nextCash = firstCashEntry;
		refCashEntry(firstCashEntry).prevCash = ci;
		cashEntry.prevCash = -1;
		firstCashEntry = ci;
	}
	return ci; 
}

/// ensure that a given box is cashed
short AbstCashedBoxFiles::ensureCashedWithoutRead (int bi)
{
	short ci = cashIndex[bi];
	// if box is not already cashed
	if (ci == -1) {
		// check if the next available cash entry was already used
		ci = lastCashEntry;
		AbstCashEntry& cashEntry = refCashEntry(ci);
		if (!cashEntry.empty()) {
			// and write if necessary
			if (debug && (cashEntry.modified == 0)) {
				PRINT("dismissed " << getShortFileName(cashEntry.bi) << " cash: ");
				int cj = firstCashEntry;
				for (int i=0; i<nrCashed; ++i, cj = refCashEntry(cj).nextCash) PRINT(refCashEntry(cj).bi << ":" << refCashEntry(cj).hits << " ");
				PRINTLN(refCashEntry(lastCashEntry).bi);
			}
			writeCashBox(lastCashEntry);
			cashIndex[cashEntry.bi] = -1;
		}

		// read the new box
		//if (!readCashBox(ci,bi)) return -1;
		InitCashBox (ci, bi);

	}

	// move entry to front
	if (ci != firstCashEntry) {
		// increment hits
		AbstCashEntry& cashEntry = refCashEntry(ci);
		++cashEntry.hits;
		// remove entry from list
		int pci = cashEntry.prevCash;
		IFWARN(pci == -1, 0, "no prev cash!!!") return ci;
		if (lastCashEntry == ci) lastCashEntry = pci;
		int nci = cashEntry.nextCash;
		refCashEntry(pci).nextCash = nci;
		if (nci != -1) refCashEntry(nci).prevCash = pci;
		// insert at front
		cashEntry.nextCash = firstCashEntry;
		refCashEntry(firstCashEntry).prevCash = ci;
		cashEntry.prevCash = -1;
		firstCashEntry = ci;
	} 
	return ci;
}



/// return the ci-th cashed entry
int AbstCashedBoxFiles::getCashedBox(int ci)
{
	int i = firstCashEntry;
	for (int j=0; j<ci; ++j) i = refCashEntry(i).nextCash;
	return refCashEntry(i).bi;
}

void AbstCashedBoxFiles::makeEmpty()
{
	for (int ci = 0; ci < nrCashed; ++ci) refCashEntry(ci).dismiss();
	for (int bi = 0; bi < nrBoxes; ++bi) cashIndex[bi] = -1;
}


/// construct the file from the total number of boxes and the maximum number of cashed boxes
AbstCashedBoxFiles::AbstCashedBoxFiles(const string& baseName, int _nrBoxes, int _nrCashed, bool _debug)
{
	nrCashed = _nrCashed;
	debug = _debug;
	/// store the base file name
	baseFileName = baseName;

	/// store the number of boxes
	nrBoxes = _nrBoxes;
	/// for each box store a short index to specify the cash index or -1 if not cashed
	int i;
	cashIndex = new short[nrBoxes];
	for (i = 0; i<nrBoxes; ++i) cashIndex[i] = -1;

	/// the cash index of the next to be used cash entry
	firstCashEntry = 0;
	lastCashEntry = nrCashed-1;
	/// the cash entries
	initCounters();
}

/// construct the file from the total number of boxes and the maximum number of cashed boxes
template <class T>
CashedBoxFiles<T>::CashedBoxFiles(const string& baseName, int _nrBoxes, int _nrCashed, bool _debug, int NeedExtraRoom) : AbstCashedBoxFiles(baseName, _nrBoxes, _nrCashed, _debug)
{
	m_NeedExtraRoom = NeedExtraRoom;
	m_Write = 1;

	if (baseName == "NULL") 
		m_Write = 0;

	/// the cash entries
	cash = new CashEntry<T>[nrCashed];
	for (int i = 0; i < nrCashed; ++i) {

		if (m_NeedExtraRoom != 0)
			cash [i].MaxSize = 10;	// dummy value. we increse max size when we read

		cash[i].nextCash = (i == lastCashEntry) ? -1 : i+1;
		cash[i].prevCash = i-1;
	}
}

template <class T>
void CashedBoxFiles<T>::showStats()
{
/*	double total = 0;
	for (int bi = 0; bi < nrBoxes; ++bi) {
		int fs = fileSize(getFileName(bi));
		if (fs > 0) {
			fs /= sizeof(T);
			total += fs;
		}
	}
	total *= CashEntry<T>::sizeScaleFactor();
	PRINTLN("Cash " << getShortFileName() << " read frac = " << readElements/total << ";  write frac = " << writtenElements/total);
	*/
	PRINTLN("Cash " << getShortFileName() << " read frac = " << (double)readBoxes/nrBoxes << ";  write frac = " << (double) writtenBoxes/nrBoxes);
}

/// return a short version of filename
string AbstCashedBoxFiles::getShortFileName() const
{
	int pos = baseFileName.find_last_of('\\');
	if (pos == string::npos) pos = 0; else ++pos;
	int pos2 = baseFileName.find_last_of(".ooc");
	if (pos2 == string::npos) pos2 = baseFileName.find_last_of(".");
	return baseFileName.substr(pos,pos2-pos);
}
/// return a short version of filename
string AbstCashedBoxFiles::getShortFileName(int bi) const
{
	int pos = baseFileName.find_last_of('\\');
	if (pos == string::npos) pos = 0; else ++pos;
	int pos2 = baseFileName.find_last_of(".ooc");
	return getShortFileName()+":"+toString(bi);
}
/// destructor
AbstCashedBoxFiles::~AbstCashedBoxFiles()
{
	delete [] cashIndex;
}
/// destructor
template <class T>
CashedBoxFiles<T>::~CashedBoxFiles()
{
	flush();
	if (debug) showStats();

	delete [] cash;
}
/// write all modified blocks to disk
void AbstCashedBoxFiles::flush()
{
	for (int ci = 0; ci < nrCashed; ++ci) writeCashBox(ci);
}

/// fill new files with empty entries*/
template <class T>
void CashedBoxFiles<T>::attach(const string& parentFileName, int parentSize)
{
	for (int bi = 0; bi < nrBoxes; ++bi) {
		string name = getOutOfCoreName(baseFileName,bi);
		if (fileExists(name)) deleteFile(name);
		int size = fileSize(getOutOfCoreName(parentFileName,bi));
		IFWARN(size == -1,0,"could not determine parent file size") return;
		CashEntry<T>::create(name, size);
	}
}
/// write a cash box of given cash index only if modified
template <class T>
bool CashedBoxSingleFile<T>::writeCashBox(int ci)
{
	// check if box was modified
	CashEntry<T>& cashEntry = cash[ci];
	if (cashEntry.modified == 1) {
		IFWARN(!cashEntry.write64(baseFileName,((__int64)cashEntry.bi)*boxSize), 0, "could not write box file " << getOutOfCoreName(baseFileName,cashEntry.bi)) 
			exit(-1);//return false;
		if (debug) PRINTLN("wrote " << ci << " block " << getShortFileName(cashEntry.bi));
		// accumulate number of written elements
		writtenElements += cashEntry.size;
		writtenBoxes += 1;
		cashEntry.modified = 0;
	}
	return true;
}
template <class T>
void CashedBoxSingleFile<T>::showStats()
{
	double total = fileSize64(baseFileName);
	if (total > 0) {
		total /= sizeof(T);
		total *= CashEntry<T>::sizeScaleFactor();
		PRINTLN("Cash " << getShortFileName() << " read frac = " << readElements/total << ";  write frac = " << writtenElements/total);
	}
}

/// read a cash box from a given file
template <class T>
bool CashedBoxSingleFile<T>::readCashBox(int ci, int bi)
{
	// check if box was modified
	CashEntry<T>& cashEntry = cash[ci];
	__int64 offset = ((__int64)bi)*boxSize;
	cashEntry.size = (bi == nrBoxes-1) ? lastBoxSize() : boxSize;

	//if (m_Write == 1) {
	IFWARN(!cashEntry.read64(baseFileName,offset,cashEntry.size), 0, "could not read box file " << getOutOfCoreName(baseFileName,cashEntry.bi)) {
		printf ("ehhhe\n");
			return false;
	}
		if (debug) PRINTLN("read in " << ci << " block " << getShortFileName(bi));
		readElements += cashEntry.size;
		++readBoxes;
		/*
	}
	else {
		cashEntry.elements = NULL;
	}
	*/

	// update cash entry and cash index field 
	cashEntry.hits     = 0;
	cashEntry.modified = 0;
	cashEntry.bi       = bi;
	cashIndex[bi]      = ci;
	return true;
}

/// construct the file from the total number of boxes and the maximum number of cashed boxes
template <class T>
CashedBoxSingleFile<T>::CashedBoxSingleFile(const string& baseName, int _nrBoxes, int _nrCashed, bool _debug) : CashedBoxFiles<T>(baseName,_nrBoxes,_nrCashed,_debug)
{
	nrElements = fileSize64(baseFileName)/sizeof(T);
	IFWARN(nrElements == -1, 0, "could not open file " << baseFileName) return;
	boxSize = (nrElements+nrBoxes-1)/nrBoxes;
	// make sure the box size is dividable by 8
	boxSize = 8*((boxSize+7)/8);
	if (boxSize >= MAX_LI) {
		boxSize = MAX_LI-7;
	}
	nrBoxes = (nrElements+boxSize-1)/boxSize;
	if (nrBoxes != _nrBoxes) {
		int i;
		delete [] cashIndex;
		cashIndex = new short[nrBoxes];
		for (i = 0; i<nrBoxes; ++i) cashIndex[i] = -1;
	}
	if (debug) PRINTLN("nr elements = " << nrElements << " nrBoxes = " << nrBoxes << " boxSize = " << boxSize << " nr cashed = " << nrCashed);
}
