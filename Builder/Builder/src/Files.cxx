#include "Files.h"

#include <stdio.h>
#include <direct.h>
#include <io.h>
#include <fcntl.h>

bool fileExists(const string& fileName)
{
	FILE* fp = ::fopen(fileName.c_str(), "r");
	if (fp) fclose(fp);
	return fp != NULL;
}

bool fileExists(const char *fileName)
{
	FILE* fp = ::fopen(fileName, "r");
	if (fp) fclose(fp);
	return fp != NULL;
}


int fileSize(const string& fileName)
{
	int fh = _open(fileName.c_str(), _O_BINARY | _O_RDONLY);
	if (fh == -1) return -1;
	long l = _filelength(fh);
	_close(fh);
	return l;
}

__int64 fileSize64(const string& fileName)
{
	int fh = _open(fileName.c_str(), _O_BINARY | _O_RDONLY);
	if (fh == -1) return -1;
	__int64 l = _filelengthi64(fh);
	_close(fh);
	return l;
}

#include <stdlib.h>
#include <process.h>
#include <windows.h>

enum PIPES { READ_HANDLE, WRITE_HANDLE }; /* Constants 0 and 1 for READ and WRITE */

// open a gzipped file as if it was a regular file
FILE* fopenGzipped(const char* filename, const char* mode)
{
	// check mode
	if (mode[0] == 'r') {
		// open input file
		FILE* gzipInput = fopen(filename, mode);
		if (!gzipInput) return NULL;

		// create the pipe
		int hPipe[2];
		IFWARN(_pipe(hPipe, 2048, ((mode[1] =='b') ? _O_BINARY : _O_TEXT) | _O_NOINHERIT) == -1, 0, "could not create pipe") return NULL;

		// duplicate stdin handle
		int hStdIn = _dup(_fileno(stdin));
		// redirect stdin to input file
		IFWARN(_dup2(_fileno(gzipInput), _fileno(stdin)) != 0, 0, "could not redirect stdin") return NULL;

		// duplicate stdout handle
		int hStdOut = _dup(_fileno(stdout));
		// redirect stdout to write end of pipe
		IFWARN(_dup2(hPipe[WRITE_HANDLE], _fileno(stdout)) != 0, 0, "could not set pipe output") return NULL;
		// close original write end of pipe
		close(hPipe[WRITE_HANDLE]);

		// redirect read end of pipe to input file
		IFWARN(_dup2(hPipe[READ_HANDLE], _fileno(gzipInput)) != 0, 0, "could not redirect input file") return NULL;
		// close original read end of pipe
		close(hPipe[READ_HANDLE]);

		// Spawn process
		HANDLE hProcess = (HANDLE) spawnlp(P_NOWAIT, "gzip", "gzip", "-d", NULL);

		// redirect stdin back into stdin
		IFWARN(_dup2(hStdIn, _fileno(stdin)) != 0, 0, "could not reconstruct stdin") return NULL;
		// redirect stdout back into stdout
		IFWARN(_dup2(hStdOut, _fileno(stdout)) != 0, 0, "could not reconstruct stdout") return NULL;

		// return input file
		return gzipInput;
	}
	else return NULL;
}

// delete a file completely
bool deleteFile(const string& fileName)
{
	return remove(fileName.c_str()) == 0;
}

struct FileInfo
{
	_finddata_t fileinfo;
	long        handle;
};

bool dirExists(const string& dirName)
{
	bool result;
	FileInfo fi;
	fi.handle = _findfirst((string(dirName)+"\\*.*").c_str(), &fi.fileinfo);

	result = (fi.handle != -1);

	// sungeui
	_findclose( fi.handle );
	return result;
}

bool mkdir(const string& dirName)
{
	return _mkdir(dirName.c_str()) == 0;
}

bool rmdir(const string& dirName)
{
	return _rmdir(dirName.c_str()) == 0;
}
//#define AVOID_SYSTEM_CACHE 
#ifdef AVOID_SYSTEM_CACHE
// read a file into a newly allocated memory block
bool readFileVoid(const string& fileName, void* dest, int nrElements, int elementSize)
{
	static DWORD pageSize = 0;
	static void* storage = 0;
	static int allocated = 0;
	if (pageSize == 0) {
		if (!GetDiskFreeSpace(NULL, NULL, &pageSize, NULL, NULL)) {
			PRINTLN("could not determine page size (" << GetLastError() << ")");
			exit(0);
		}
	}
	HANDLE file = CreateFile(fileName.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
	if (file == INVALID_HANDLE_VALUE) {
		PRINTLN("could not open " << fileName << "(" << GetLastError() << ")");
		return false;
	}
	DWORD nr = nrElements*elementSize;
	DWORD elementsRead;
	DWORD nrParam = (DWORD) pow(2,(int)(log(nr)/log(2)+1));
	if (nrParam < pageSize) nrParam = pageSize;
	if (nrParam > allocated) {
		if (storage) {
			if (!VirtualFree(storage,allocated,MEM_DECOMMIT)) {
				PRINTLN("could not free (" << GetLastError() << ")");
			}
		}
		allocated = nrParam;
		storage = VirtualAlloc(NULL,allocated,MEM_COMMIT,PAGE_READWRITE);
		if (storage == NULL) {
			PRINTLN("could not allocate (" << GetLastError() << ")");
			return false;
		}
	}
	if (!ReadFile(file, storage, nrParam, &elementsRead, NULL)) {
		PRINTLN("could not read " << fileName << "(" << GetLastError() << ")");
		return false;
	}
	if (elementsRead != nr) {
		PRINTLN("could not read all " << "(" << GetLastError() << ")");
		return false;
	}
	memcpy(dest,storage,nr);
	if (!CloseHandle(file)) {
		PRINTLN("could not close " << fileName << "(" << GetLastError() << ")");
		return false;
	}
	return true;
}
#else
// read a file into a newly allocated memory block
bool readFileVoid(const string& fileName, void* dest, int nrElements, int elementSize)
{
	FILE* fp = fopen(fileName.c_str(), "rb");
	if (!fp) return false;
	int nrRead = fread(dest, elementSize, nrElements, fp);
	fclose(fp);
	return nrRead == nrElements;
}
#endif
/// read a block of a file into a given memory block
bool readFileVoid(const string& fileName, void* dest, unsigned long offset, int nrElements, int elementSize)
{
	FILE* fp = fopen(fileName.c_str(), "rb");
	if (!fp) return false;
	int nrRead = 0;
	unsigned long real_offset = offset *elementSize;
	
	int GB= 1024*1024*1024;
	if (real_offset < GB) {
		if (fseek(fp,real_offset,SEEK_SET) == 0 ) nrRead = fread(dest, elementSize, nrElements, fp);
	}
	else {
		fseek(fp,GB,SEEK_SET);

		real_offset -= GB;

		while (real_offset > GB) {
			fseek(fp,GB,SEEK_CUR);
			real_offset -= GB;
		}

		if (fseek(fp,real_offset,SEEK_CUR) == 0 ) nrRead = fread(dest, elementSize, nrElements, fp);
	}
	fclose(fp);
	return nrRead == nrElements;
}
/// read a block of a file into a given memory block
bool readFileVoid64(const string& fileName, void* dest, __int64 offset, int nrElements, int elementSize)
{
	FILE* fp = fopen(fileName.c_str(), "rb");
	if (!fp) return false;
	int nrRead = 0;
	__int64 real_offset = offset *elementSize;
	
	if (_fseeki64(fp,real_offset,SEEK_SET) == 0 ) nrRead = fread(dest, elementSize, nrElements, fp);
	
	fclose(fp);
	return nrRead == nrElements;
}
// write a file from a given memory block and element count and return whether this succeeded
bool writeFileVoid(const string& fileName, void* elements, int nrElements, int elementSize)
{
	FILE* fp = fopen(fileName.c_str(), "wb");
	if (!fp) 
		return false;
	int nrWritten = fwrite(elements, elementSize, nrElements, fp);
	fclose(fp);
	return nrWritten == nrElements;
}
// write a file from a given memory block and element count and return whether this succeeded
bool writeFileVoid(const string& fileName, void* elements, int offset, int nrElements, int elementSize)
{
	FILE* fp = fopen(fileName.c_str(), "r+b");
	if (!fp) return false;
	int nrWritten = 0;
	if (fseek(fp,offset*elementSize,SEEK_SET) == 0 ) nrWritten = fwrite(elements, elementSize, nrElements, fp);
	fclose(fp);
	return nrWritten == nrElements;
}
// write a file from a given memory block and element count and return whether this succeeded
bool writeFileVoid64(const string& fileName, void* elements, __int64 offset, int nrElements, int elementSize)
{
	FILE* fp = fopen(fileName.c_str(), "r+b");
	if (!fp) return false;
	int nrWritten = 0;
	if (_fseeki64(fp,offset*elementSize,SEEK_SET) == 0 ) nrWritten = fwrite(elements, elementSize, nrElements, fp);
	fclose(fp);
	return nrWritten == nrElements;
}
// append a given memory block of element count elements to a given file and return whether this succeeded
bool appendFileVoid(const string& fileName, void* elements, int nrElements, int elementSize)
{
	FILE* fp = fopen(fileName.c_str(), "ab");
	if (!fp) return false;
	int nrWritten = fwrite(elements, elementSize, nrElements, fp);
	fclose(fp);
	return nrWritten == nrElements;
}
// copy file
bool copyFile(const string &srcFileName, const string &dstFileName)
{
	FILE *fpSrc = fopen(srcFileName.c_str(), "rb");
	if(!fpSrc) return false;

	FILE *fpDst = fopen(dstFileName.c_str(), "wb");
	if(!fpDst)
	{
		fclose(fpSrc);
		return false;
	}

	unsigned char buf[1024];
	while(size_t size = fread(buf, 1, 1024, fpSrc))
		fwrite(buf, 1, size, fpDst);

	fclose(fpSrc);
	fclose(fpDst);
}