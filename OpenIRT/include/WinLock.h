#pragma once

class WinLock
{
public:
	WinLock(LPSECURITY_ATTRIBUTES lpMutexAttributes = NULL, BOOL bInitialOwner = FALSE, LPCTSTR lpName = NULL)
	{
		m_hObject = ::CreateMutex(lpMutexAttributes, bInitialOwner, lpName);
	}

	~WinLock()
	{
		if(!m_hObject) ::CloseHandle(m_hObject);
		m_hObject = NULL;
	}

	void lock(DWORD dwTimeout = INFINITE)
	{
		::WaitForSingleObject(m_hObject, dwTimeout);
	}

	void unlock()
	{
		::ReleaseMutex(m_hObject);
	}

protected:
	HANDLE m_hObject;
};