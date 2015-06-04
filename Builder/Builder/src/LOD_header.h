#ifndef LOD_header_H
#define LOD_header_H
// sungeui
// Dec-14, 2005
// LOD structures and its related headers for ray-tracing

#define SET_FIRST_TRI_IDX(Idx, i1, i2, i3) {Idx |= i1; Idx |= (i2 << 2); Idx |= (i3 << 4);}
#define SET_SECOND_TRI_IDX(Idx, i1, i2, i3) {Idx |= i1 << 6; Idx |= (i2 << 8); Idx |= (i3 << 10);}


 
typedef struct LODNode_t {
	float	m_ErrBnd;			// Area of circle that is projected 
									// from a sphere containing the node
									// This data should be in the kd-tree node

	Vector3 m_n;						// (geometric) normal vector (normalized)
	float m_d;				// d from plane equation
	unsigned char  m_i1, m_i2;	// planes to be projected to	
#ifdef _USE_TRI_MATERIALS
	unsigned short   m_material;
#endif
	
	float m_Proj_ErrBnd;			// only radius of the sphere containing the node
	//float dummy;				// To meet power_of_two constraints of OOCFile
	float m_ExtBB;				// extent of the Box; used for allowance of plane and ray intersection
} LODNode, *LODNodePtr;

class CExtLOD : public LODNode
{
public:

	Vector3 m_ShadeN;			    // Shading normal (normalized)
	unsigned int m_TriStartIdx;		// Starting idx from two triangles

	Vector3 m_Corner [4];			// 4 vertices of the quad consisting of two triangles
									// Initial format of v[4]:
									//			v [0  - v [2]					max
									//			 |	  /		|	
									//			v [1] -  v[3]		min
									//	tri 1 : v[0], v[1], v[2]
									//	tri 2 : v[2], v[1], v[3]
									// Two triangles share same d, i1, i2
	int m_TriIdx;					// each 2bit (total 3 and 3 elements) index on m_Corner

	//	Vector3 m_Center;				// Center of the plane
	//	Vector3 m_S, m_T;				// two vector defining a plane, m_Center + (m_S + m_T) define a far coner of the LOD plane
};
/*
typedef struct PreLODNode_t : public LODNode {
	Vec3f m_Min, m_Max;		// bounding box that conver all the geometry contained in the voxels.

} PreLODNode, *PreLODNodePtr;
*/


// m_NearPlane:m_PixelSize == DistToObject:MaximumAllowance
class CLODMetric
{
public:
	float m_NearPlane;	// world size
	float m_PixelSize;	// world size

	float m_PoE;		// image space size

	CLODMetric (void)
	{
		m_NearPlane = m_PixelSize = m_PoE = 1;

	}
	float ComputeMaxAllowanceModifier (void)
	{
		return m_PixelSize / m_NearPlane;

	}

};
#endif
