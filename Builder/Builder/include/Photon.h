#ifndef COMMON_PHOTON_H
#define COMMON_PHOTON_H
/********************************************************************
	created:	2009/03/26
	filename: 	Photon.h
	file base:	Photon
	file ext:	h
	author:		Kim Tae-Joon (tjkim@tclab.kaist.ac.kr)
	
	purpose:	Photon class for use in ray tracer and geometry representation
*********************************************************************/
#include "Vector2.h"
#include "Vector3.h"
#include "rgb.h"

/**
* Photon structure.
*/
typedef union Photon_t { // 12 + 2 + 2 + 8 + 8 = 32
	struct
	{
		Vector3 pos;		// position
		unsigned char phi, theta;	// incident angle (spherical coords)
		short plane;		// plane flag for tree
		unsigned short power[4];	// Light power(2 byte for each R,G,B,A)
		unsigned int left;
		unsigned int right;
	};
	struct
	{
		Vector3 pos;		// position
		unsigned char phi, theta;	// incident angle (spherical coords)
		short plane;		// plane flag for tree
		unsigned short power[4];	// Light power(2 byte for each R,G,B,A)
		unsigned int children;
		unsigned int children2;
	};
} Photon, *PhotonPtr;

#endif