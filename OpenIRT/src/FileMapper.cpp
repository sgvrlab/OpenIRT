#include "FileMapper.h"

stdext::hash_map<void *, std::pair<HANDLE, HANDLE> > FileMapper::s_handles;
