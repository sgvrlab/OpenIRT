#ifndef COMMON_COMMON_H_
#define COMMON_COMMON_H_

/********************************************************************
	created:	2003/10/26
	created:	26:10:2003   15:45
	filename: 	Common\common.h
	file path:	Common
	file base:	common
	file ext:	h
	author:		Christian Lauterbach
	
	purpose:	Common declarations and types
*********************************************************************/

#include <iostream>
#include <vector>


//////////////////////////////////////////////////////////////////////////
// Engine features compile-time options
// (comment to disable)
//


#define _TIMER_OUTPUT_		// add timing codes, which dump times into files


// SCALE issue: it's hard to find one rule that is dependent on model size.

#define BUNNY			// BB size : 10
//#define DRAGON
//#define ST_MATTHEW		// BB size : 1000		PP, Lucy
//#define FOREST			// BB size : 10K

// enable LOD
//#define USE_LOD
//#define WORKING_SET_COMPUTATION		// compute working set and cache misses

//#define NO_BACKFACE_CULLING		// for CAD model like PP


// Compile for loading scene data in an out-of-core manner
// This enables loading larger scenes in the OOC file format, but support
// for all other scene formats (PLY, VRML) will be dropped.
#define _USE_OOC

//#define _USE_RACBVH

//#define _USE_RACM

//
// Parallel options:
//

// use OpenMP parallelization. (will only be used if supported by compiler)
// automatically selects as many threads as there are CPUs in the system.
#define _USE_OPENMP

#define NUM_THREADS 1

#define NUM_LOCK_SECTIONS 256

#define _COMPRESS_TRI
#define _USE_TRI_DELTA_ENCODING

#ifdef _USE_RACBVH
//#define _VICTIM_POLICY_SECOND_CHANCE
#define _VICTIM_POLICY_RANDOM
#endif

// Compile to load the scene from .ooc files, but instead of using OOCFile,
// the file will be mapped into memory in one big chunk. This only works 
// as long as the scene is smaller than the available address space. This
// is the preferred mode for 64-bit processors and 64-bit OS.
//#define _USE_OOC_DIRECTMM

// selects whether to use
#define OOC_TRI_FILECLASS		OOCFile64
#define OOC_VERTEX_FILECLASS	OOCFile
#define OOC_LOD_FILECLASS		OOCFile
#ifdef _USE_RACBVH
#define OOC_BSPNODE_FILECLASS	RACBVH
#else
#define OOC_BSPNODE_FILECLASS	OOCFile64
#endif
#define OOC_BSPIDX_FILECLASS	OOCFile64
#define USE_OOC_FILE_LRU
//#define USE_OOC_FILE_MOD

// use extended 16-byte kD-tree nodes with two child offsets
// and a LOD index instead of the compressed 8-byte nodes
#define KDTREENODE_16BYTES

// use extended 16-byte BVH nodes with two child offsets
// and a LOD index instead of the compressed 8-byte nodes
#define BVHNODE_16BYTES

// to deal with more than 512M nodes 
// work with KDTREENODE_16BYTES
//#define FOUR_BYTE_FOR_KD_NODE	

// if disabled, triangles cannot have individual materials
//#define _USE_TRI_MATERIALS


#define _USE_FAKE_BASEPLANE

// enable/disable eflections
#define _USE_REFLECTIONS
// enable/disable refractions
//#define _USE_REFRACTIONS
// enable/disable shadow rays for ray tracing and direct illumination sampling for path tracing
#define _USE_SHADOW_RAYS

#define _USE_AREA_LIGHT

// Disable/enable interpolated vertex normals for shading. All shading will use the
// normal surface normal. This may greatly effect the look of "curved" surfaces
// such as spheres approximated by triangles, but will save some more memory
// on each triangle.
//#define _USE_VERTEX_NORMALS

#ifndef _USE_OOC
// Stuff that does not work with OOC since the triangle layout is fixed:

// Disable/enable interpolated vertex normals for shading. All shading will use the
// normal surface normal. This may greatly effect the look of "curved" surfaces
// such as spheres approximated by triangles, but will save some more memory
// on each triangle.
#define _USE_VERTEX_NORMALS

// Disable/enable support for texturing. Will save memory since no more texture coordinates
// need to be saved for each triangle.
//#define _USE_TEXTURING
#endif

// Texture options:
//
// Kind of filtering to use for bitmap texture access. Point filtering is faster but 
// looks bad if magnified, especially in combination with environment mapping. Bilinear filtering
// is slower but not as pixelated - though it may blur details.
#define TEXTURE_FILTER_POINT 0
#define TEXTURE_FILTER_BILINEAR 1

#define TEXTURE_FILTER TEXTURE_FILTER_POINT

// Shadow/Light options:
//
// Insert light sources into any PLY scene at random positions instead of
// one light source at a fixe position. If enabled, then further options 
// can be configured below
//#define PLY_RANDOM_LIGHTS 

// enable stochastic (distributed) ray tracing for area light sources
// NOTE: if enabled, *only* area lights will be used, point light sources
// in the scene are ignored! Set number of shadow rays in options.xml.
//#define _USE_AREA_SHADOWS


//
// Global illumination options:
//

// Compile as path tracer
//#define _USE_PATH_TRACING

// Use photon map for all visualization
//#define _USE_PHOTON_MAP

// Use histogram grid algorithm instead of photon mapping (requires _USE_PHOTON_MAP)
//#define PHOTON_MAP_USE_GRID

//
// SIMD Options:
//

// Use the SIMD ray-tracing versions which can trace 4 rays in parallel ?
// (CPU needs SSE1 support !). If disabled, the legacy BSP tree will be
// used that traces one ray at a time.
//#define _USE_SIMD_RAYTRACING

// all pixels in image where the secondary hits are found to be 
// incoherent are shown in red (e.g. along object boundaries)
//#define _SIMD_SHOW_COHERENCY

// generate and show extra statistics for SIMD tracing, such as
// overhead for coherent tracing. only works when OpenMP is disabled!
//#define _SIMD_SHOW_STATISTICS

// Use beam tracing mode to exploit spatial coherence.
//#define _USE_BEAM_TRACING 

#define BEAM_PRIMARY_STARTSIZE 32
#define BEAM_SUBDIVIDE_FACTOR  4

//
///////////////////////////////////////////////////////////////////////////

/************************************************************************/
/* ´ö¼ö, Collision Detection                                            */
/************************************************************************/
//#define DO_COLLISION_DETECTION
/************************************************************************/


// Compiler and platform detection:
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
//#		define _CRT_SECURE_NO_DEPRECATE
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

// if OpenMP not supported by compiler, undefine the option
// for compiling the parallel ray tracer:
//#ifndef _OPENMP
//#undef _USE_OPENMP
//#endif

// if parallel version, include the OpenMP header
#ifdef _USE_OPENMP

#include <omp.h>
#else
// otherwise:
// simulate a few OpenMP functions needed in the code
FORCEINLINE int omp_get_max_threads() { return 1; }
FORCEINLINE int omp_get_num_threads() { return 1; }
FORCEINLINE int omp_get_thread_num() { return 0; }
FORCEINLINE void omp_set_num_threads() {}
#endif

// Hierarchy type, KD_TREE, BVH ...
#define TYPE_KD_TREE 1
#define TYPE_BVH 2
#define HIERARCHY_TYPE TYPE_BVH
#define NOT_IMPLEMENTED

// stuff for hashing with char* strings.
#include <hash_map>
#include <hash_set>
class stringhasher : public stdext::hash_compare < char* > {
public:
	size_t operator() (const char* __s) const
	{
		unsigned long __h = 0;		
		for ( ; *__s; ++__s)
			__h = 5*__h + *__s;		
		return size_t(__h);
	}

	bool operator() (const char* s1, const char* s2) const
	{
		int retval = strcmp(s1,s2);
		if(retval < 0)
			return true;
		return false;
	}

};
using stdext::hash_map; 
using stdext::hash_set;

// Helper functions
#include "helpers.h"

// Common types
#include "Types.h"

// Includes for several necessary classes
#include "OptionManager.h"
#include "Logger.h"

// Math helpers
#include "Math.h"

// Geometric types
#include "Geometry.h"

// Use the fluid studios memory manager for finding memory leaks ?
// (will create memory.log and memleaks.log in working directory)
//#include "mmgr.h"  

#endif // COMMON_COMMON_H_