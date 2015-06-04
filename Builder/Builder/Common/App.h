// Programmer: Sung-eui Yoon


#define STDEXT          // Turn on to be compiled at .net 2005
                          // Turn off to be compiled with VC++ 6.0

#ifndef STDEXT
  #define USE_STL_Hashmap    // turn on if you want to use stlport in VC++ ver 6.0
#endif


//#define DEBUG_CODEC     // enable extra codec info to verify correctness
//#define DEBUG_MODE      // enable extra debug routine

#define PREPROCESS      // indicate preprocess
                        // comment out, if the application is runtime

//#define RUNTIME		// use MemoryMap for accessing data
//#define TET_MESH	// dealing with test meshes
			// otherwise, it is triangle mesh

//#define QUAD_MESH	// make quad mesh given triangle mesh format

//#define _TIMER_OUTPUT_      // data logging
//#define WORKING_SET         // compute working set


//#define PARTIAL_MODELER_UPDATE    // for faster encoding and decoding with range encoder

#define EXTRAFAST           // faster RC of martin
//#define PETER_RCODER        // enable/disable Peter's Range coder


