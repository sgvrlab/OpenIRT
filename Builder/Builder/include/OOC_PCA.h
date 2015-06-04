#ifndef OOC_PCA_H
#define OOC_PCA_H

//	Sung-Eui Yoon
//	Dec-13, 2005 near the SIGGRAPH deadline

#include "rgb.h"
#include "Vector3.h"

class COOCPCAwoExtent
{
private:
	bool ConsiderArea;		// consider area of the triangle
	float m_NumData;		// sum of weight
	int m_ContainedTriangle;

	// for position, geometric PC including normal
	Vector3 m_Sum;
	Vector3 m_Sym;		// xx, yy, zz
	Vector3 m_Asym;		// xy, yz, zx

	// for shading normal
	Vector3 m_SumNormal;
	float m_Elevation, m_Azimuth;
	rgb m_SumColor;

	Vector3 m_Center;		// computed center of geometry
	Vector3 m_Axis [3];	// computed PCs

	float m_Eigen [3];		// store eigenvalues
							// later, compute surface deviation in object space
	float m_Extent [3];		// extent of kd-node

public:

	COOCPCAwoExtent (void) {
		ConsiderArea = true;
		m_NumData = 0;

		m_Sum = m_Sym = m_Asym = m_SumNormal = m_Center = Vector3 (0, 0, 0);
		m_Axis [0] = m_Axis [0] = m_Axis [0] = Vector3 (0, 0, 0);
		m_SumColor = rgb (0, 0, 0);

		m_Elevation = m_Azimuth = 0;
		m_ContainedTriangle = 0;
	}

	bool IsEmpty (void) 
	{
		if (m_ContainedTriangle == 0)
			return true;
		return false;
	}

	bool InsertTriangle (Vector3 V [3], const Vector3 & Normal, int index = 0);
	bool InsertVertex (Vector3 & v, const Vector3 & n, float weight);
	bool ComputePC (Vector3 & Center, Vector3 Extent [3]);
	//bool ComputePC (Vector3 & Center, Vector3 Extent [3], float &SurfaceDeviation);
	Vector3 GetShadingNormal (void);
	Vector3 GetGeometricNormal (void);
	bool Get4Corners (Vector3 [4], Vector3 * RealBB);
	rgb GetMeanColor (void);
	rgb GetQuantizedMeanColor (void);
	int GetNumContainedTriangle (void)
	{
		return m_ContainedTriangle;
	}

	// build a bigger cluster given two clusters
	COOCPCAwoExtent operator + (const COOCPCAwoExtent & A) const  
	{ 
		COOCPCAwoExtent Sum;

		Sum.m_NumData = m_NumData + A.m_NumData;
		Sum.m_Sum = m_Sum + A.m_Sum;
		Sum.m_Sym = m_Sym + A.m_Sym;
		Sum.m_Asym = m_Asym + A.m_Asym;

		Sum.m_SumNormal = m_SumNormal + A.m_SumNormal;
		Sum.m_SumColor = m_SumColor + A.m_SumColor;

		Sum.m_ContainedTriangle = m_ContainedTriangle + A.m_ContainedTriangle;

		return (Sum); 
	};


		
};



#endif
