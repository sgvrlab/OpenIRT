#include "stdafx.h"
#include "AABBTreeNode.h"

extern int markUpdateMaterial;

float AABBTreeNode::construct(int depth, ModelInstance *object, TriangleIndexList &triIDs) {	
	// only one triangle left, make this a leaf:
	if (triIDs.size() == 1){
		leafTriID = (triIDs[0] << 2) | 3;		
		Triangle &tri = GETTRI(object, triIDs[0]);	
		_bbox.SetBB(GETVERTEX(object, tri.p[0]));
		_bbox.Update(GETVERTEX(object, tri.p[1]));
		_bbox.Update(GETVERTEX(object, tri.p[2]));

		#ifdef MARK_UPDATED_NODES
		tri.material = markUpdateMaterial;
		#endif

		return _bbox.surfaceArea();
	}	

	// build bounding box:
	_bbox.Reset();
	for (unsigned int index = 0; index < triIDs.size(); index++) {
		const Triangle &tri = GETTRI(object, triIDs[index]);	
		_bbox.Update(GETVERTEX(object, tri.p[0]));
		_bbox.Update(GETVERTEX(object, tri.p[1]));
		_bbox.Update(GETVERTEX(object, tri.p[2]));
	}

	// find biggest axis:
	Vector3 diff = _bbox.GetBBMax() - _bbox.GetBBMin();
	int biggestaxis = diff.indexOfMaxComponent();
	float split_pt = .5 * diff[biggestaxis] +_bbox.GetBBMin()[biggestaxis];

	// compute average primitve location:
	float avgloc;
	TriangleIndexList Lchildren;
	TriangleIndexList Rchildren;
	for (unsigned int index = 0; index < triIDs.size(); index++) {
		const Triangle &tri = GETTRI(object, triIDs[index]);

		avgloc = GETVERTEX(object, tri.p[0]).e[biggestaxis]; 
		avgloc += GETVERTEX(object, tri.p[1]).e[biggestaxis]; 
		avgloc += GETVERTEX(object, tri.p[2]).e[biggestaxis]; 
		avgloc *=.33333333334;

		if (avgloc < split_pt) {			
			Lchildren.push_back(triIDs[index]);
		} else {
			Rchildren.push_back(triIDs[index]);
		}
	}

	int lsz = Lchildren.size();
	int rsz = Rchildren.size();	

	// special case: subdivision did not work out, just go half/half
	if (lsz ==0 || rsz ==0) {
		Lchildren.clear();
		Rchildren.clear();
		for (unsigned int index=0; index < triIDs.size()/2; index++) {
			Lchildren.push_back(triIDs[index]);
		}
		for (unsigned int index = triIDs.size()/2; index < triIDs.size(); index++) {
			Rchildren.push_back(triIDs[index]);
		}
		lsz = Lchildren.size();
		rsz = Rchildren.size();
	}

	// allocate children:
	lChild = (AABBTreeNode *)malloc(2 * sizeof(AABBTreeNode));

	// construct children recursively
	object->treeStats.numNodes+=2;

	// surface areas:
	float childSA = getLeftChild()->construct(depth+1, object, Lchildren)
				  + getRightChild()->construct(depth+1, object, Rchildren);	
	float mySA = _bbox.surfaceArea();

	// find axis with the least overlap and set for traversal ordering
	Vector3 childBBOverlap = getLeftChild()->_bbox.GetBBMax() - getRightChild()->_bbox.GetBBMin();
	leafTriID |= childBBOverlap.indexOfMinComponent();	

	saRatio = mySA / childSA;

	return mySA;
}


float AABBTreeNode::update_indexed_threaded(ModelInstance *object, float &SARatioIncrease) {
	int threads_backup = omp_get_max_threads();
	omp_set_num_threads(4);
	float returns[4];
	float ratios[4];
	ratios[0] = 0.0f;
	ratios[1] = 0.0f;
	ratios[2] = 0.0f;
	ratios[3] = 0.0f;	

	#ifdef _USE_OPENMP
	#pragma omp parallel sections
	#endif
	{
		#ifdef _USE_OPENMP
		#pragma omp section
		#endif
		{
			returns[0] = getLeftChild()->getLeftChild()->updateBB_indexed(object, ratios[0]);
		}
		#ifdef _USE_OPENMP
		#pragma omp section
		#endif
		{
			returns[1] = getLeftChild()->getRightChild()->updateBB_indexed(object, ratios[1]);		
		}
		#ifdef _USE_OPENMP
		#pragma omp section
		#endif
		{
			returns[2] = getRightChild()->getLeftChild()->updateBB_indexed(object, ratios[2]);
		}
		#ifdef _USE_OPENMP
		#pragma omp section
		#endif
		{
			returns[3] = getRightChild()->getRightChild()->updateBB_indexed(object, ratios[3]);
		}
	}

	getRightChild()->_bbox.SetBB(getRightChild()->getRightChild()->_bbox.GetBBMin());
	getRightChild()->_bbox.Update(getRightChild()->getRightChild()->_bbox.GetBBMax());
	getRightChild()->_bbox.Update(getRightChild()->getLeftChild()->_bbox.GetBBMin());
	getRightChild()->_bbox.Update(getRightChild()->getLeftChild()->_bbox.GetBBMax());

	getLeftChild()->_bbox.SetBB(getLeftChild()->getRightChild()->_bbox.GetBBMin());
	getLeftChild()->_bbox.Update(getLeftChild()->getRightChild()->_bbox.GetBBMax());
	getLeftChild()->_bbox.Update(getLeftChild()->getLeftChild()->_bbox.GetBBMin());
	getLeftChild()->_bbox.Update(getLeftChild()->getLeftChild()->_bbox.GetBBMax());

	_bbox.SetBB(getLeftChild()->_bbox.GetBBMin());
	_bbox.Update(getLeftChild()->_bbox.GetBBMax());
	_bbox.Update(getRightChild()->_bbox.GetBBMin());
	_bbox.Update(getRightChild()->_bbox.GetBBMax());

	float leftNewSA = getLeftChild()->_bbox.surfaceArea();
	float rightNewSA = getRightChild()->_bbox.surfaceArea();

	float leftSARatio = leftNewSA / (returns[0] + returns[1]);
	float rightSARatio = rightNewSA / (returns[2] + returns[3]);

	SARatioIncrease = ratios[0] + ratios[1] + ratios[2] + ratios[3] + (leftSARatio - getLeftChild()->saRatio) + (rightSARatio - getRightChild()->saRatio);

	omp_set_num_threads(threads_backup);

	return (returns[0] + returns[1] + returns[2] + returns[3]);
} 

float AABBTreeNode::updateBB_indexed(ModelInstance *object, float &SARatioIncrease) {	
	
	if (isLeaf()) {
		Triangle &tri = GETTRI(object, getTriID());	
		_bbox.SetBB(GETVERTEX(object, tri.p[0]));
		_bbox.Update(GETVERTEX(object, tri.p[1]));
		_bbox.Update(GETVERTEX(object, tri.p[2]));	

		#ifdef MARK_UPDATED_NODES
		if (tri.material == markUpdateMaterial)
			tri.material = object->materiallist.size() - 1;
		#endif

		return _bbox.surfaceArea();
	}
	else { // inner leaf
		float newChildSA = getLeftChild()->updateBB_indexed(object, SARatioIncrease);
		_bbox.SetBB(getLeftChild()->_bbox.GetBBMin(), getLeftChild()->_bbox.GetBBMax());
		newChildSA += getRightChild()->updateBB_indexed(object, SARatioIncrease);
		_bbox.UpdateMax(getRightChild()->_bbox.GetBBMax());
		_bbox.UpdateMin(getRightChild()->_bbox.GetBBMin());

		float myNewSA = _bbox.surfaceArea();
		
		// find axis with the least overlap and set for traversal ordering
		Vector3 childBBOverlap = getLeftChild()->_bbox.GetBBMax() - getRightChild()->_bbox.GetBBMin();
		leafTriID = (leafTriID & ~3) | childBBOverlap.indexOfMinComponent();

		float newSARatio = myNewSA / newChildSA;		
		SARatioIncrease += newSARatio - saRatio;

		return myNewSA;
	}
}


void AABBTreeNode::destroy() {
	if (!isLeaf()) {
		getLeftChild()->destroy();
		getRightChild()->destroy();
		free(getLeftChild());
	}
}

void AABBTreeNode::destroyAccumulate(TriangleIndexList &triIDs) {
	if (!isLeaf()) {
		getLeftChild()->destroyAccumulate(triIDs);
		getRightChild()->destroyAccumulate(triIDs);
		free(getLeftChild());
	}
	else {
		triIDs.push_back(getTriID());
	}
}
