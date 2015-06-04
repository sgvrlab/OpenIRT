#ifndef _ANN_H
#define _ANN_H

//#include "MyString.h"
#include <fstream>
#include <ostream>
#include <ann/ann.h>
#include <vec3f.hpp>

namespace YOON
{

class CANN
{
	bool m_IsReady;
	int m_NumVertices;

	/// the number of nearest neighbors
	int m_ann_k;
	/// the precision for nearest neighbors
	double m_ann_eps;

	ANNpointArray	m_data_pts;		// data points
	ANNpoint	m_query_pt;		// query point
	ANNidxArray	m_nn_idx;			// near neighbor indices
	ANNdistArray	m_dists;			// near neighbor distances
	ANNkd_tree	* m_pTree;		// search structure

public:

	CANN (void);
	~CANN (void);
	
	void SetNumVertices (int NumVertex);
	void SetGeomVertex (int Index, Vec3f & Ver);	
	int GetApproximateNNs (Vec3f & Input, int * pOutput);

};


}	// end of namespace
#endif

