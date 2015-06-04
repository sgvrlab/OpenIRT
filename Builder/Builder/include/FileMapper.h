#ifndef FILE_MAPPER_HPP
#define FILE_MAPPER_HPP
/********************************************************************
	created:	2013/01/24
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	FileMapper
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	For easy usage of windows memory mapped file
*********************************************************************/

#include <Windows.h>
#include <stdio.h>
#include <hash_map>

class FileMapper 
{
protected:
	static stdext::hash_map<void *, std::pair<HANDLE, HANDLE> > s_handles;

public:
	static void *map(const char *fileName, bool isRead = true)
	{
		BY_HANDLE_FILE_INFORMATION fileInfo;

		DWORD fileMappingMode;
		DWORD fileShareMode;
		DWORD fileOpenMode;

		if(isRead)
		{
			fileMappingMode = GENERIC_READ;
			fileShareMode = FILE_SHARE_READ;
			fileOpenMode = OPEN_EXISTING;
		}
		else 
		{
			fileMappingMode = GENERIC_WRITE | GENERIC_READ;
			fileShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
			fileOpenMode = CREATE_NEW;
		}

		HANDLE hFile, hMapping;

		// open file:
		if (! (hFile = CreateFile(fileName, fileMappingMode, fileShareMode, NULL, fileOpenMode,
			FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL))) 
		{
				printf("Cannot open file: %s\n", fileName);			
				exit (-1);
		}

		// get file size:
		GetFileInformationByHandle(hFile, &fileInfo);

		ULARGE_INTEGER fileSize;

		fileSize.LowPart = fileInfo.nFileSizeLow; 
		fileSize.HighPart = fileInfo.nFileSizeHigh;

		if(isRead)
			fileMappingMode = PAGE_READONLY;
		else
			fileMappingMode = PAGE_READWRITE;

		if (!(hMapping = CreateFileMapping(hFile, NULL, fileMappingMode, 
			fileSize.HighPart, fileSize.LowPart, NULL))) 
		{
				printf("CreateFileMapping() failed\n");
				exit (-1);
		}

		void *data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);

		s_handles[data] = std::pair<HANDLE, HANDLE>(hFile, hMapping);

		return data;
	}

	static void unmap(void *address)
	{
		if (!UnmapViewOfFile(address)) 
		{
			printf("UnmapViewOfFile(%X) failed\n", address);
		}

		CloseHandle(s_handles[address].second);
		CloseHandle(s_handles[address].first);
	}
};

#endif