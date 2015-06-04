#ifndef COMMON_VERTEX_H
#define COMMON_VERTEX_H
/********************************************************************
	created:	2009/03/05
	filename: 	Vertex.h
	file base:	Vertex
	file ext:	h
	author:		Kim Tae-Joon (tjkim@tclab.kaist.ac.kr)
	
	purpose:	Vertex class for use in ray tracer and geometry representation
*********************************************************************/
#include "Vector2.h"
#include "Vector3.h"

/**
 * Main vertex structure for OOC use
 */
typedef struct Vertex_t {
	Vector3 v;				// vertex geometry
	Vector3 n;				// normal vector
	Vector3 c;				// color
	Vector2 uv;				// Texture coordinate
	unsigned char dummy[20];
} Vertex, *VertexPtr;

typedef struct VertexV_t {
	Vector3 v;				// vertex geometry
	unsigned char dummy[4];
} VertexV, *VertexVPtr;

typedef struct VertexVN_t {
	Vector3 v;				// vertex geometry
	Vector3 n;				// normal vector
	unsigned char dummy[8];
} VertexVN, *VertexVNPtr;

typedef struct VertexVC_t {
	Vector3 v;				// vertex geometry
	Vector3 c;				// color
	unsigned char dummy[8];
} VertexVC, *VertexVCPtr;

typedef struct VertexVT_t {
	Vector3 v;				// vertex geometry
	Vector2 uv;				// Texture coordinate
	unsigned char dummy[12];
} VertexVT, *VertexVTPtr;

typedef struct VertexVNC_t {
	Vector3 v;				// vertex geometry
	Vector3 n;				// normal vector
	Vector3 c;				// color
	unsigned char dummy[28];
} VertexVNC, *VertexVNCPtr;

typedef struct VertexVCT_t {
	Vector3 v;				// vertex geometry
	Vector3 c;				// color
	Vector2 uv;				// Texture coordinate
} VertexVCT, *VertexVCTPtr;

typedef struct VertexVNT_t {
	Vector3 v;				// vertex geometry
	Vector3 n;				// normal vector
	Vector2 uv;				// Texture coordinate
} VertexVNT, *VertexVNTPtr;

typedef struct VertexVNCT_t {
	Vector3 v;				// vertex geometry
	Vector3 n;				// normal vector
	Vector3 c;				// color
	Vector2 uv;				// Texture coordinate
	unsigned char dummy[20];
} VertexVNCT, *VertexVNCTPtr;

typedef enum VertexType_t {
	V, VN, VC, VT, VNC, VCT, VNT, VNCT
} VertexType;

#endif