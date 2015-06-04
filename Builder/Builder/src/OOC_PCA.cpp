#include "OOC_PCA.h"

#ifdef PI
#undef PI
#undef HALF_PI
#endif

#include "Magic/MgcEigen.h"

#define CONVERT_Vector3_VECTOR3(s,d) {d.e [0] = s.x;d.e [1] = s.y; d.e[2] = s.z;}
#define CONVERT_VECTOR3_Vector3(s,d) {d.x = s.e [0];;d.y = s.e [1];d.z = s.e [2];}


#define DEGREE_89	0.01745

#ifndef PI
#define PI		3.141592f	// Pi
#endif


bool COOCPCAwoExtent::InsertTriangle (Vector3 v [3], const Vector3 & Normal, int index)
{
	Vector3 & v1 = v [0];
	Vector3 & v2 = v [1];
	Vector3 & v3 = v [2];

	float Area = cross(v2 - v1, v3 - v1).length() * 0.5f;
	Area *= 10000000;

	if (Area == 0)		// degenerated case
		return false;

	if (!(Area > 0.0f && Area < FLT_MAX))
	{
		printf("Area = %f\n", Area);
	}

	if (!(Normal.e[0] > -FLT_MAX && Normal.e[0] < FLT_MAX))
	{
		printf("Normal = %f %f %f\n", Normal.e[0], Normal.e[1], Normal.e[2]);
	}

	assert (Area > 0);

	// color summation
	//m_SumColor += (Area*Color);
	
	int i;
	for (i = 0;i < 3;i++)
		InsertVertex (v [i], Normal, Area/3.);

	m_ContainedTriangle++;

	return true;
}

// insert a vertex with weight
// compute independent component to compute PC
bool COOCPCAwoExtent::InsertVertex (Vector3 & v, const Vector3 & n, float weight)
{
	m_NumData += weight;
	
	// for geometry, geometry normal
	m_Sum += (weight*v);
	m_Sym += (weight*v&v);

	Vector3 Asym (v[1], v[2], v[0]);
	m_Asym += (weight*v&Asym);

	// for shading normal
	m_SumNormal += (weight*n);

	//float AbsN = n.Length ();
	
	/*
	// convert normal into unit sphere, having two components
	// Step1: project the vector into z = 0;
	Vector3 Nor_Pro_Z = n;
	Nor_Pro_Z.z = 0;
	Nor_Pro_Z.normalize ();

	// Step2: compute two components
	Vector3 PivotV = Vector3 (1, 0, 0);
	m_Azimuth += (weight * acos (PivotV*Nor_Pro_Z));
	m_Elevation += (weight * acos (n*Nor_Pro_Z));
	*/

	return true;
}

Vector3 COOCPCAwoExtent::GetGeometricNormal (void)
{
	Vector3 Normal = m_SumNormal;
	Normal.normalize ();
	return Normal;
}

Vector3 COOCPCAwoExtent::GetShadingNormal (void)
{
	Vector3 Normal = m_SumNormal;
	Normal.normalize ();
	return Normal;
}

// If we want to compute a plane from PCA,
// Extent [2] is a geometric normal of data
bool COOCPCAwoExtent::ComputePC (Vector3 & Center, Vector3 Extent [3])
{
	// build covariance matrix

	Vector3 Mean = m_Sum / m_NumData;
	m_Center = Center = Mean;

	// compute eigenvectors for covariance, code from Mgc
	//Vector3 m_Asym;		// xy, yz, zx
	Mgc::Eigen kES(3);
	kES.Matrix(0,0) = m_Sym [0]/m_NumData - Mean [0]*Mean [0];	// fSumXX
	kES.Matrix(0,1) = m_Asym [0]/m_NumData - Mean[0]*Mean [1];	// fSumXY
	kES.Matrix(0,2) = m_Asym [2]/m_NumData - Mean[0]*Mean [2];	// fSumXZ;
	kES.Matrix(1,0) = kES.Matrix(0,1);
	kES.Matrix(1,1) = m_Sym [1]/m_NumData - Mean [1]*Mean [1];		// fSumYY
	kES.Matrix(1,2) = m_Asym [1]/m_NumData - Mean[1]*Mean [2];	// fSumYZ;
	kES.Matrix(2,0) = kES.Matrix(0,2);
	kES.Matrix(2,1) = kES.Matrix(0,2);
	kES.Matrix(2,2) = m_Sym [2]/m_NumData - Mean [2]*Mean [2];	// fSumZZ;
	kES.IncrSortEigenStuff3();

	

	Extent[0].e[0] = kES.GetEigenvector(0,0);
	Extent[0].e[1] = kES.GetEigenvector(1,0);
	Extent[0].e[2] = kES.GetEigenvector(2,0);
	Extent[1].e[0] = kES.GetEigenvector(0,1);
	Extent[1].e[1] = kES.GetEigenvector(1,1);
	Extent[1].e[2] = kES.GetEigenvector(2,1);
	Extent[2].e[0] = kES.GetEigenvector(0,2);
	Extent[2].e[1] = kES.GetEigenvector(1,2);
	Extent[2].e[2] = kES.GetEigenvector(2,2);

	/*
	// measure surface deviation
	float SumEigenValue = kES.GetEigenvalue (0) + kES.GetEigenvalue (1) + kES.GetEigenvalue (2);
	//float SurfaceVariance = kES.GetEigenvalue (0) / SumEigenValue;
	float SurfaceVariance = kES.GetEigenvalue (0) / kES.GetEigenvalue (1);

	float EV0 = kES.GetEigenvalue (0) * Extent [0].Length ();
	float EV1 = kES.GetEigenvalue (1) * Extent [1].Length ();
	float EV2 = kES.GetEigenvalue (2) * Extent [2].Length ();
	*/

	int i;
	for ( i = 0 ; i < 3 ; i++ )
		m_Eigen [i] = kES.GetEigenvalue (i);

	//printf ("Surface deviation = %f\n", SurfaceVariance * 100);
	//SurfaceDeviation = SurfaceVariance;
//	printf ("Surface deviation = %f\n", EV0);


	// compute a normal that is close to shading normal.
	Vector3 ShadingN = m_SumNormal;
	ShadingN = ShadingN.normalize ();

	/*
	float m_a = m_Azimuth/m_NumData;
	float m_e = m_Elevation/m_NumData;

	Vector3 ShadingN;
	ShadingN.x = sin(m_e)*cos(m_a);
	ShadingN.y = sin(m_e)*sin(m_a);
	ShadingN.z = cos(m_e);
	ShadingN.normalize ();
	*/


	// this is wrong
	//Vector3 GeometryN = Extent [0].normalize (); // smallest variance (0th vector) , close to geometric normal

	 ///*
	// compute an eigenvector, which is closest to the Shading normal.
	int NormalIdx = 0;
	float MaxCosTheta = -1;
	float CosTheta;
	for (i = 0;i < 3;i++)
	{
		 CosTheta = dot (ShadingN, Extent [i].normalize ());
		if (CosTheta < 0)
			CosTheta = dot (ShadingN, - Extent [i]);

		if (CosTheta > MaxCosTheta) {
			MaxCosTheta = CosTheta;
			NormalIdx = i;
		}
	}

	/*
	if (NormalIdx != 0)
		printf ("New eigen vector = %d (%f)\n", NormalIdx, CosTheta);
	*/

	// swapping a normal eigenvector to become first eigenvector
	Vector3 tempV = Extent [0];
	Extent [0] = Extent [NormalIdx];
	Extent [NormalIdx] = tempV;

	float temp = m_Eigen [0];
	m_Eigen [0] = m_Eigen [NormalIdx];
	m_Eigen [NormalIdx] = temp;
	
	Vector3 GeometryN = Extent [0];
	//*/

	CosTheta = dot (ShadingN, GeometryN);
	if (CosTheta < DEGREE_89) 	// bigger than 89 degree 
		Extent [0] = - Extent [0];
	
	for (i = 0;i < 3;i++)
	{
		m_Axis [i] = Extent [i];		
		m_Axis [i].normalize ();
	}

	// for constant coordinate when we compute 4 corners and keep same norma betwee
	// m_Axis [0] and a computed normal from 4 corners
	Vector3 ProjectedZ = cross(m_Axis [1], m_Axis [2]);

	//float AngBtwVectors = acos (dot (ProjectedZ, m_Axis [0]));
	CosTheta = dot (ProjectedZ, m_Axis [0]);		// numerically more stable
	
	//if (AngBtwVectors < - PI2 / 6. || AngBtwVectors > PI2 / 6.) {	// bigger than 60 degree 
	if (CosTheta < DEGREE_89) {	// bigger than 89 degree 
		Vector3 temp = m_Axis [2];
		m_Axis [2] = m_Axis [1];
		m_Axis [1] = temp;
	}

	float t1 = dot (m_Axis [0], m_Axis [1]);
	float t2 = dot (m_Axis [0], m_Axis [2]);
	float t3 = dot (m_Axis [1], m_Axis [2]);

	return true;
}

// compute 2 triangles on the simplification plane.
// Triangles shoud cover original geometry. To achieve this, we project each vertex
// of bbox of the node into the triangle
// format of v[4]:
//			v [0  - v [2]					max
//			 |	  /		|	
//			v [1] -  v[3]		min
//	tri 1 : v[0], v[1], v[2]
//	tri 2 : v[2], v[1], v[3]

bool COOCPCAwoExtent::Get4Corners (Vector3 V [4], Vector3 * RealBB)
{
	// I got codes from Magic
	// Let C be the box center and let U0, U1, and U2 be the box axes.  Each
    // input point is of the form X = C + y0*U0 + y1*U1 + y2*U2.  The
    // following code computes min(y0), max(y0), min(y1), max(y1), min(y2),
    // and max(y2).  The box center is then adjusted to be
    //   C' = C + 0.5*(min(y0)+max(y0))*U0 + 0.5*(min(y1)+max(y1))*U1 +
    //        0.5*(min(y2)+max(y2))*U2
	int i;
	Vector3 Min, Max;
	//CONVERT_VECTOR3_Vector3(_Min,Min);
	//CONVERT_VECTOR3_Vector3(_Max,Max);

	// Project RealBB to compute conservative corners
	Min = RealBB [0];
	Max = RealBB [1];

	//assert (Min.x <= _Min.x () && Min.y <= _Min.y () && Min.z <= _Min.z ());
	//assert (Max.x >= _Max.x () && Max.y <= _Max.y () && Max.z <= _Max.z ());

	/*
	if (Min.x > _Min.x ())
		Min.x = _Min.x ();
	if (Min.y > _Min.y ())
		Min.y = _Min.y ();
	if (Min.z > _Min.z ())
		Min.z = _Min.z ();

	if (Max.x < _Max.x ())
		Max.x = _Max.x ();
	if (Max.y < _Max.y ())
		Max.y = _Max.y ();
	if (Max.z < _Max.z ())
		Max.z = _Max.z ();
	*/


	Vector3 BBV [8];

	// CALCULATE VERTICES OF BOX
	#define SET(v,x,y,z) v.e[0]=x; v.e[1]=y; v.e[2]=z;
	SET(BBV[0],Max[0],Max[1],Max[2]); SET(BBV[4],Max[0],Max[1],Min[2]);
	SET(BBV[1],Min[0],Max[1],Max[2]); SET(BBV[5],Min[0],Max[1],Min[2]);
	SET(BBV[2],Min[0],Min[1],Max[2]); SET(BBV[6],Min[0],Min[1],Min[2]);
	SET(BBV[3],Max[0],Min[1],Max[2]); SET(BBV[7],Max[0],Min[1],Min[2]);


    Vector3 kDiff = BBV [0] - m_Center;
    float fY0Min = dot(kDiff, m_Axis [0]), fY0Max = fY0Min;
    float fY1Min = dot(kDiff, m_Axis [1]), fY1Max = fY1Min;
    float fY2Min = dot(kDiff, m_Axis [2]), fY2Max = fY2Min;

    for (i = 1; i < 8; i++)
    {
        kDiff = BBV[i] - m_Center;

        float fY0 = dot(kDiff, m_Axis [0]);
        if ( fY0 < fY0Min )
            fY0Min = fY0;
        else if ( fY0 > fY0Max )
            fY0Max = fY0;

        float fY1 = dot(kDiff, m_Axis [1]);
        if ( fY1 < fY1Min )
            fY1Min = fY1;
        else if ( fY1 > fY1Max )
            fY1Max = fY1;

        float fY2 = dot(kDiff, m_Axis[2]);
        if ( fY2 < fY2Min )
            fY2Min = fY2;
        else if ( fY2 > fY2Max )
            fY2Max = fY2;
    }

	/*
    Vector3 _newCenter = m_Center + 
		(0.5f*(fY0Min+fY0Max))*m_Axis [0] +
        (0.5f*(fY1Min+fY1Max))*m_Axis [1] +
        (0.5f*(fY2Min+fY2Max))*m_Axis [2];
	*/

	//float newExtent [3];
	/*
	newExtent [0] = 0.5f*(fY0Max - fY0Min);
    newExtent [1] = 0.5f*(fY1Max - fY1Min);
    newExtent [2] = 0.5f*(fY2Max - fY2Min);
	*/

	m_Extent [0] = (fY0Max > fabs(fY0Min)) ? fY0Max : fabs (fY0Min);
	m_Extent [1] = (fY1Max > fabs(fY1Min)) ? fY1Max : fabs (fY1Min);
	m_Extent [2] = (fY2Max > fabs(fY2Min)) ? fY2Max : fabs (fY2Min);


	//			v [0  - v [2]					max
	//			 |	  /		|	
	//			v [1] -  v[3]		min
	V [0] = m_Center - m_Extent [1]*m_Axis [1] + m_Extent [2]*m_Axis [2];
	V [1] = m_Center - m_Extent [1]*m_Axis [1] - m_Extent [2]*m_Axis [2];
	V [2] = m_Center + m_Extent [1]*m_Axis [1] + m_Extent [2]*m_Axis [2];
	V [3] = m_Center + m_Extent [1]*m_Axis [1] - m_Extent [2]*m_Axis [2];


	// capture surface deviation
	//SurfaceDeviation = newExtent [0];

	//SET_FIRST_TRI_IDX (Idx, 0, 1, 2);
	//SET_SECOND_TRI_IDX (Idx, 2, 1, 3);

	return true;
}

rgb COOCPCAwoExtent::GetMeanColor (void)
{
	return m_SumColor / m_NumData;
}

// only allow 10 variance for each components, so, we can have 1000 different color.
// Later, we need to pick orthogonal one so that we can have more compact and more colors.

rgb COOCPCAwoExtent::GetQuantizedMeanColor (void)
{
	const float QuantizationLevel = 25;
	rgb M_C = m_SumColor / m_NumData;

	int i;
	for (i = 0;i < 3;i++)
		M_C.data [i] = floor (M_C.data [i] * QuantizationLevel) / QuantizationLevel;

	return M_C;
}
