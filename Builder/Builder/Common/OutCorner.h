// Programmer: Sung-eui Yoon


#ifndef _Out_Corner_H
#define _Out_Corner_H

#include "App.h"
#include "vec3f.hpp"
//#include "IndexPair.h"
#include "assert.h"

const unsigned int MAX_UINT = (unsigned int) (pow ((float) 2,(float) 32) - 1);
//#define unsigned int IndexPair;

#ifdef TET_MESH
const int NUM_TRI_ELE	= 4;	// tet mesh
#else
const int NUM_TRI_ELE	= 3;	// triangle mesh
#endif

class CGeomVertex : public Vec3f
{
public :
	#ifdef NORMAL_INCLUDED
	Vec3f m_Normal;
	#endif 
	#ifdef COLOR_INCLUDED
	Color3 m_Color;
	#endif

	#ifdef TET_MESH
	  float	m_Value;
	#else 	  
	  unsigned int m_c;	 // a corner that shares this vertex 
	#endif 


	bool InDiagonalEdge (CGeomVertex V);
};

class COutCorner
{
public :
	unsigned int m_v;	
	unsigned int m_NextCorner;		// next corner for triangle
						// opposite corner for Tet	
};

class COutTriangle {
public :
	COutCorner m_c [NUM_TRI_ELE]; // corners for this triangle 

  /*
	#ifndef TET_MESH
	// for triangle mesh
	COutCorner m_Dummy;	// to align in 4K cache block;
	#endif
  */

	bool setCorner(const int WhichC, const int CornerIdx)
	{
		assert (WhichC >= 0 && WhichC < NUM_TRI_ELE); 
		m_c [WhichC].m_NextCorner = CornerIdx;
		return true;
	}

	void getAllOppCorners (unsigned int *_c);
	// Return corner in neighboring tet opp. given corner number.
	int getOppCorner(const int C);

};

#endif
