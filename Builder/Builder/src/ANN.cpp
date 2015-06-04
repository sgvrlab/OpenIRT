#include "ANN.h"
#include "assert.h"

using namespace YOON;

CANN::CANN (void)
{

	// the number of nearest neighbors
	m_ann_k = 7;			// -> get m_ann_k + 1 neighbor
	// the precision for nearest neighbors
	m_ann_eps = 0.01;

	m_IsReady = false;
}

CANN::~CANN (void)
{
	if (m_IsReady == true) {
		delete m_pTree;
		delete [] m_dists;
		delete [] m_nn_idx;
		annDeallocPts (m_data_pts);   // deallocate data points
		annDeallocPt (m_query_pt);
	}
}

void CANN::SetNumVertices (int NumVertex)
{
	m_NumVertices = NumVertex;

	m_data_pts  = annAllocPts (NumVertex, 3);   // allocate and copy data points
}

void CANN::SetGeomVertex (int Index, Vec3f & Ver)
{
	m_data_pts [Index][0] = Ver.x;
	m_data_pts [Index][1] = Ver.y;
	m_data_pts [Index][2] = Ver.z;

}

int CANN::GetApproximateNNs (Vec3f & Input, int * pOutput)
{
	if (m_IsReady == false) {
		m_query_pt = annAllocPt (3);			// allocate query point
		m_nn_idx   = new ANNidx [m_ann_k+1];			// allocate near neigh indices
		m_dists    = new ANNdist [m_ann_k+1];			// allocate near neighbor dists
		
		m_pTree = new ANNkd_tree (			// build search structure
				m_data_pts,			// the data points
				m_NumVertices,			// number of points
				3);			// dimension of space


		m_IsReady = true;
	}

	if (m_ann_k + 1 > m_NumVertices) {
		int i;
		for (i = 0;i < m_NumVertices;i++)
			pOutput [i] = i;

		return m_NumVertices;
	}

	m_query_pt [0] = Input.x;
	m_query_pt [1] = Input.y;
	m_query_pt [2] = Input.z;

	m_pTree->annkSearch(			// search
			m_query_pt,		// query point
			m_ann_k+1,		// number of near neighbors
			m_nn_idx,			// nearest neighbors (returned)
     			m_dists,			// distance (returned)
			m_ann_eps);			// error bound

	int i;
	for (i = 0;i <= m_ann_k;i++)
	{
		assert (m_nn_idx [i] < m_NumVertices && m_nn_idx [i] >= 0);
 
		pOutput [i] = m_nn_idx [i];		
	}

	return m_ann_k + 1;

}
