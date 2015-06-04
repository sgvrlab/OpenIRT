#ifndef COMMON_H
#define COMMON_H

// OOC Options:

//#define DE_MODEL	// need to enable it for DE_MODEL --> color change

//#define USE_QUANTIZED_COLOR		

//#define _USE_VERTEX_NORMALS


// sungeui start ------------------------------------
// for OOC use; this flag should be always turned on.
#define _USE_OOC
// sungeui end --------------------------------------

//#define _USE_LOD

#ifdef _USE_LOD
#define _USE_CONTI_NODE
#define _USE_ONE_TRI_PER_LEAF
#endif

//#define _USE_VOXEL_OUTPUT

// use OpenMP parallelization (will only be used if supported by compiler)
#define _USE_OPENMP

// use a certain number of threads, comment out to use automatic determination
// based on number of processors
//#define NUM_OMP_THREADS 2

// use triangle materials:
#define _USE_TRI_MATERIALS

// use extended size kD-tree nodes (for LOD):
#define KDTREENODE_16BYTES

// use extended size BVH nodes (for LOD):
#define BVHNODE_16BYTES

//#define FOUR_BYTE_FOR_KD_NODE		// to deal with more than 512M nodes 
//#define FOUR_BYTE_FOR_BV_NODE			// to deal with more than 512M nodes 

// use memory debugging for leaks (also include mmgr.c into project and rebuild!):
//#include "mmgr.h"


// selects whether to use
#define OOC_TRI_FILECLASS		OOCFile64
#define OOC_VERTEX_FILECLASS	OOCFile64
#define OOC_LOD_FILECLASS		OOCFile64
#define OOC_BSPNODE_FILECLASS	OOCFile64
#define OOC_BSPIDX_FILECLASS	OOCFile64
#define OOC_NodeLayout_FILECLASS	OOCFile64



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
#ifndef _OPENMP
#undef _USE_OPENMP
#endif

// if parallel version, include the OpenMP header

#ifdef _USE_OPENMP
#include <omp.h>
#else
// otherwise:
// simulate a few OpenMP functions needed in the code
FORCEINLINE int omp_get_max_threads() { return 1; }
FORCEINLINE int omp_get_num_threads() { return 1; }
FORCEINLINE int omp_get_thread_num() { return 0; }
FORCEINLINE void omp_set_num_threads(int num) {}
#endif

// Hierarchy type, KD_TREE, BVH ...
#define TYPE_KD_TREE 1
#define TYPE_BVH 2
#define HIERARCHY_TYPE TYPE_BVH
#endif