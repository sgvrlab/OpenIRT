/********************************************************************
	created:	2009/04/24
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	defines
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Common definitions
*********************************************************************/

#pragma once

//
// Compile time options 
//


//
// Common defines
//
#ifndef MAX_PATH
#define MAX_PATH 260
#endif

#define MAX_NUM_THREADS 32

#include <vector>
typedef std::vector<std::string> FileList;
typedef FileList::iterator FileListIterator;

//
// Selected menu
//
enum SelectedMenu
{
	MENU_FULLSCREEN,
	MENU_SCREENSHOT,
	MENU_UNLOAD_SCENE,
	MENU_SAVE_CAMERA,
	MENU_QUIT,

	MENU_RASTERIZATION,
	MENU_RAY_TRACING,
	MENU_PATH_TRACING,
	MENU_PHOTON_MAPPING,
	MENU_INSTANT_RADIOSITY,

	MENU_DRAW_ALL_BB,
	MENU_DRAW_BB_DXD,

	MENU_LOAD_SCENE
};

//
// Rendering mode
//
enum RenderingMode
{
	RENDER_MODE_CPU_RAY_TRACING,
	RENDER_MODE_RASTERIZATION,
	RENDER_MODE_RAY_TRACING,
	RENDER_MODE_PATH_TRACING,
	RENDER_MODE_PHOTON_MAPPING,
	RENDER_MODE_TREX,
	RENDER_MODE_DEBUGGING
};

//
// Compiler and platform detection
//
#define PLATFORM_WIN32 1
#define PLATFORM_LINUX 2
#define PLATFORM_APPLE 3

#define COMPILER_MSVC 1
#define COMPILER_GNUC 2
#define COMPILER_BORL 3

#if defined( _MSC_VER )
#   define COMMON_COMPILER COMPILER_MSVC
#   define COMMON_COMP_VER _MSC_VER

#elif defined( __GNUC__ )
#   define COMMON_COMPILER COMPILER_GNUC
#   define COMMON_COMP_VER __VERSION__

#elif defined( __BORLANDC__ )
#   define COMMON_COMPILER COMPILER_BORL
#   define COMMON_COMP_VER __BCPLUSPLUS__

#else
#   pragma error "No known compiler. Abort! Abort!"
#endif

/* See if we can use __forceinline or if we need to use __inline instead */
#if COMMON_COMPILER == COMPILER_MSVC 
#   if COMMON_COMP_VER >= 1200
#		ifndef _CRT_SECURE_NO_DEPRECATE
#		define _CRT_SECURE_NO_DEPRECATE
#		endif
#       define FORCEINLINE __forceinline
#   endif
#else
#   define FORCEINLINE __inline
#endif

/* Finds the current platform */
#if defined( __WIN32__ ) || defined( _WIN32 )
#   define COMMON_PLATFORM PLATFORM_WIN32
#elif defined( __APPLE_CC__)
#   define COMMON_PLATFORM PLATFORM_APPLE
#else
#   define COMMON_PLATFORM PLATFORM_LINUX
#endif

// UNIX specific headers
#if COMMON_PLATFORM == PLATFORM_LINUX || COMMON_PLATFORM == PLATFORM_APPLE
#include <sys/time.h>
#include <unistd.h>
#endif

// Parallel options
#ifdef _OPENMP
#include <omp.h>
#else
// simulate a few OpenMP functions needed in the code
FORCEINLINE int omp_get_max_threads() { return 1; }
FORCEINLINE int omp_get_num_threads() { return 1; }
FORCEINLINE int omp_get_thread_num() { return 0; }
FORCEINLINE void omp_set_num_threads(int i) {}
#endif

#define _mm_dot3_ps(x1, y1, z1, x2, y2, z2) ( _mm_add_ps(_mm_add_ps(_mm_mul_ps((x1), (x2)), _mm_mul_ps((y1), (y2))), _mm_mul_ps((z1), (z2))) )

/**
 * 4-bit mask to 128-bit mask for SSE registers/instructions
 **/
__declspec(align(16)) static const unsigned int maskLUTable[16][4] = { 
	{ 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	// 0
	{ 0xffffffff, 0x00000000, 0x00000000, 0x00000000 }, // 1
	{ 0x00000000, 0xffffffff, 0x00000000, 0x00000000 }, // 2
	{ 0xffffffff, 0xffffffff, 0x00000000, 0x00000000 }, // 3
	{ 0x00000000, 0x00000000, 0xffffffff, 0x00000000 }, // 4
	{ 0xffffffff, 0x00000000, 0xffffffff, 0x00000000 }, // 5
	{ 0x00000000, 0xffffffff, 0xffffffff, 0x00000000 }, // 6
	{ 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000 }, // 7
	{ 0x00000000, 0x00000000, 0x00000000, 0xffffffff },	// 8
	{ 0xffffffff, 0x00000000, 0x00000000, 0xffffffff }, // 9
	{ 0x00000000, 0xffffffff, 0x00000000, 0xffffffff }, // 10
	{ 0xffffffff, 0xffffffff, 0x00000000, 0xffffffff }, // 11
	{ 0x00000000, 0x00000000, 0xffffffff, 0xffffffff }, // 12
	{ 0xffffffff, 0x00000000, 0xffffffff, 0xffffffff }, // 13
	{ 0x00000000, 0xffffffff, 0xffffffff, 0xffffffff }, // 14
	{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff }, // 15
};

//#define TEST_VOXEL