#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#include <ctype.h>

/**
 * Converts a C-style string to lowercase
 */
inline void strtolower(char *str) {
	while (*str) {
		*str = tolower(*str);
		str++;
	}
}

/**
 * Converts a C-style string to uppercase
 */
inline void strtoupper(char *str) {
	while (*str) {
		*str = tolower(*str);
		str++;
	}
}

/**
 * Performance/time functions. Since timers must
 * be queried differently in Win32 and UNIX, this
 * is different for each platform
 */

#if COMMON_PLATFORM == PLATFORM_WIN32
#include <windows.h>

class TimerValue {
public:
	TimerValue() {
		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&value);
		elapsed.QuadPart = 0;
	}

	// copy constructor:
	TimerValue(const TimerValue& orig) {
		value = orig.value;
		frequency = orig.frequency;
	}

	void set() {
		QueryPerformanceCounter(&value);
	}

	double diff(TimerValue &other) {
		return (double)(other.value.QuadPart - value.QuadPart) / (double)frequency.QuadPart;
	}

	void addDiff(TimerValue &other) {
		elapsed.QuadPart += other.value.QuadPart - value.QuadPart;
	}

	void addDiffFromNow() {
		LARGE_INTEGER temp;
		QueryPerformanceCounter(&temp);
		elapsed.QuadPart += temp.QuadPart - value.QuadPart;
	}

	double getElapsed() {
		return (double)(elapsed.QuadPart) / (double)frequency.QuadPart;
	}

	friend float operator-(const TimerValue &, const TimerValue &);

protected:
	LARGE_INTEGER value;
	LARGE_INTEGER frequency;
	LARGE_INTEGER elapsed;
};

/**
 * Counts CPU cycles instead of time, so suitable for extremely high accuracy
 * (and also very fast)
 */
class CycleCounter {
public:
	CycleCounter() {
		elapsed = 0;
		set();
	}

	// copy constructor:
	CycleCounter(const CycleCounter& orig) {
		elapsed = orig.elapsed;
		startTime = orig.startTime;
	}

	void set() {
		startTime = getCycles();
	}

	void addDiffFromNow() {
		elapsed += getCycles() - startTime;
	}

	unsigned __int64 getElapsed() {
		return elapsed;
	}

protected:

	ULONGLONG getCycles() {
		UINT32 timehi = 0, timelo = 0;

		/*
		__asm
		{
			rdtsc				// query CPU counter
			mov timehi, edx;    // high DWORD
			mov timelo, eax;    // low DWORD
		}
		*/

		return ((ULONGLONG)timehi << 32) + (ULONGLONG)timelo;
	}

	ULONGLONG elapsed;
	ULONGLONG startTime;
};

inline float operator-(const TimerValue &v1, const TimerValue &v2) {
	return (float)(v1.value.QuadPart - v2.value.QuadPart) / (float)v1.frequency.QuadPart;
}
#endif


#ifdef _USE_OOC_DIRECTMM

void* allocateFullMemoryMap(const char * pFileName);
bool deallocateFullMemoryMap(void *address);

#endif // _USE_OOC_DIRECTMM

#endif