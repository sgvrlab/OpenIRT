#include "stdafx.h"
#include "common.h"

using namespace std;

typedef struct {
	HANDLE fileHandle;		// handle of open file
	HANDLE mmHandle;		// handle of mapping
	void *mappingAddress;	// address of mapped memory
	LARGE_INTEGER fileSize; // size of file (64 bit)
} memoryMapping;

typedef stdext::hash_map< unsigned int, memoryMapping > MMTable;
typedef MMTable::iterator MMTableIterator;
MMTable memoryMappings;

void outputWindowsErrorMessage() {
	DWORD  ErrorCode = GetLastError();
	LPVOID lpMsgBuf;

	FormatMessage ( FORMAT_MESSAGE_ALLOCATE_BUFFER | 
		FORMAT_MESSAGE_FROM_SYSTEM | 
		FORMAT_MESSAGE_IGNORE_INSERTS, 0, ErrorCode, 0, // Default language
		(LPTSTR) &lpMsgBuf,   0,   NULL );

	std::cerr << (char *)lpMsgBuf << std::endl;
}

void* allocateFullMemoryMap(const char * pFileName) {	
	//SYSTEM_INFO systemInfo;
	BY_HANDLE_FILE_INFORMATION fileInfo;	
	memoryMapping newMapping;

	//char output[200];

	// open file:
	if (! (newMapping.fileHandle = CreateFile(pFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL))) {
		std::cerr << "allocateFullMemoryMap(): Cannot open file: " << pFileName << std::endl;			
		outputWindowsErrorMessage();
		exit(-1);
	}

	// get file size:
	GetFileInformationByHandle(newMapping.fileHandle, &fileInfo);
	newMapping.fileSize.LowPart = fileInfo.nFileSizeLow; 
	newMapping.fileSize.HighPart = fileInfo.nFileSizeHigh;

	if (!(newMapping.mmHandle = CreateFileMapping(newMapping.fileHandle, NULL, PAGE_READONLY, newMapping.fileSize.HighPart, newMapping.fileSize.LowPart, NULL))) {
		std::cerr << "allocateFullMemoryMap(): CreateFileMapping() failed" << std::endl;
		outputWindowsErrorMessage();
		exit (-1);
	}

	// map the whole file to memory:
	if (!(newMapping.mappingAddress = (void *)MapViewOfFile(newMapping.mmHandle, FILE_MAP_READ, 0,0,0))) {
		std::cerr << "MapViewOfFile() failed:" << std::endl;
		outputWindowsErrorMessage();
		exit (-1);
	}

	cout << "allocateFullMemoryMap(" << pFileName << "): " << (newMapping.fileSize.QuadPart / (__int64)1024) << " KB mapped.\n";

	// enter new mapping into list:
	memoryMappings[(unsigned int)newMapping.mappingAddress] = newMapping;
	return newMapping.mappingAddress;
}

bool deallocateFullMemoryMap(void *address) {
	char output[200];
	MMTableIterator it = memoryMappings.find((unsigned int)address);

	if (it != memoryMappings.end()) {
		memoryMapping &mmEntry = it->second;

		if (!UnmapViewOfFile(address)) {
			std::cerr << "UnmapViewOfFile(" << (unsigned int)address << ") failed:" << std::endl;
			outputWindowsErrorMessage();
		}

		CloseHandle(mmEntry.mmHandle);
		CloseHandle(mmEntry.fileHandle);
		return true;
	}
	else {
		printf("deallocateFullMemoryMap(): address %x not in table.\n", address);		
		return false;
	}
}