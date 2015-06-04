#ifndef COMMON_MODELINSTANCE_H
#define COMMON_MODELINSTANCE_H

#include <float.h>

#if defined(_USE_OOC) && !defined(_USE_OOC_DIRECTMM)


# include "OOCFile.h"
# include "OOCFile64.h"
#ifdef _USE_RACM
# include "RACBVH.h"
#endif
#ifdef _USE_RACM
#include "compressed_mesh.h"
#endif
//<<<<<<< .mine
/*
typedef OOC_TRI_FILECLASS<Triangle>*			 SceneTriangleList;
typedef OOC_VERTEX_FILECLASS<_Vector4>*			 SceneVertexList;
*/

//=======
#include "mem_manager.h"

//typedef OOC_TRI_FILECLASS<Triangle>*			 SceneTriangleList;
//typedef OOC_VERTEX_FILECLASS<_Vector4>*			 SceneVertexList;
typedef CMemManager <Triangle>*			 SceneTriangleList;
typedef CMemManager <_Vector4>*			 SceneVertexList;


//>>>>>>> .r2405

typedef OOC_LOD_FILECLASS<LODNode>*				 SceneLodList;
typedef OOC_BSPNODE_FILECLASS<BSPArrayTreeNode>* BSPNodeList;
typedef OOC_BSPIDX_FILECLASS<unsigned int>*      BSPIndexList;

#else

typedef Triangle*			SceneTriangleList;	
typedef _Vector4*			SceneVertexList;
typedef LODNode*			SceneLodList;
typedef BSPArrayTreeNode*	BSPNodeList;
typedef unsigned int*		BSPIndexList;

#endif

typedef Vector3*			VertexNormalList;

// macros for accessing single triangles and vertices, depending on
// whether we're using OOC or normal mode
#if defined(_USE_OOC) && !defined(_USE_OOC_DIRECTMM)
#define MODEL_GETTRI(idx) (*(trilist))[idx]
//#define MODEL_GETVERTEX(idx) (*(vertices))[idx*sizeof(_Vector4)]
#define MODEL_GETVERTEX(idx) (*(vertices))[idx]
#else
#define MODEL_GETTRI(idx) trilist[idx]
#define MODEL_GETVERTEX(idx) vertices[idx]
#endif

#define MODEL_GETMATERIAL(materialid) (materiallist[materialid])


/**
 *	Instance of a model in the scene. Defines geometry
 *  references, materials & further properties
 */
class BVH;

class ModelInstance {
public:

	//
	// Geometry information (may be instanced)
	//
	bool useRACM;

	SceneTriangleList trilist;		// Triangles	
	SceneVertexList   vertices;		// Vertices
	SceneLodList	  lodlist;		// LOD structures
	BSPNodeList		  tree;			// kd-tree nodes
	BSPIndexList	  indexlist;	// kd-tree triangle indices 
	MaterialList	  materiallist; // list of materials
	#ifdef _USE_RACM
	CMeshAbstract		*pRACM;
	#endif
	//VertexNormalList  vertexNormals;

	#if HIERARCHY_TYPE == TYPE_BVH
	// BSP tree of the scene
	BVH *bvh;
	#endif
	
	float *g_QuanErrs;				// LOD error quantification table

	//
	// Transformation and scene placement:
	//
	
	Vector3 bb[2];					// axis-aligned bounding box (world coords)
	Vector3 translate_world;		// Translation world-coord -> model coord (equals start position)
	Vector3 translate_world_start;	// Translation world-coord -> model coord (equals start position)

	Vector3 transformedBB[2];		// bounding box transformed.


	Vector3 translate_animate;		// Translation for animation (values per second)
	unsigned int animationFrames;	// number of frames for animation until looped
	float currentAnimationFrame;


	// test - start
	unsigned int sizeTransformMatList;		// size of transform matrixes
	unsigned int indexCurrentTransform;		// current index in tramsform matrix list
	Matrix* transformMatList;				// transform matrix list
	Matrix* invTransformMatList;			// inverse transform matrix list	
	// test - end

	unsigned int ID;				// id of the model
	
	// model stats:
	unsigned int nVerts;			// number of vertices
	unsigned int nTris;				// number of triangles
	unsigned int nLODs;				// number of LODs

	BSPTreeInfo  treeStats;			// detailed tree stats

	bool isInstanced;				// is this model instanced (i.e. references another model?)
	bool boundingBoxGenerated;		// has working bounding box

	char i1, i2;					// main axis for texture coord faking
	float coordToU, coordToV;		// transformation factor to multiply relative coordinate
	                                // with to yield the texture coordinate
	float tiling;

	char *modelFileName;			// file name of model file corresponding to this instance

	// 
	// useful methods:
	//

	void transformObj(const Matrix& tranMat);

	// set all values to zero:
	ModelInstance() {
		useRACM = false;
		#ifdef _USE_RACM
		pRACM = NULL;
		#endif
		OptionManager *opt = OptionManager::getSingletonPtr();
		memset(this, 0, sizeof(ModelInstance));
		bb[0] = Vector3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
		bb[1] = Vector3(FLT_MAX,FLT_MAX,FLT_MAX);
		tiling = opt->getOptionAsFloat("raytracing", "generatedTexCoordsTiling", 1.0);

		transformMatList = NULL;
		invTransformMatList = NULL;	

		ID = -1;
	}

	~ModelInstance() {
		destroy();
	}

	// make this an instance of another model:
	void instanceFrom(ModelInstance *other);

	void animate(float timeDifference);

	// deallocate arrays:
	void destroy() {

		// when we're instanced, do not deallocate the data arrays
		// because the real model will do that.
		if (!isInstanced) {
		
			// deallocate geometry data
			#if defined(_USE_OOC_DIRECTMM) && defined(_USE_OOC)
				if (vertices)
					deallocateFullMemoryMap(vertices);
				if (trilist)
					deallocateFullMemoryMap(trilist);
			#else
				if(!useRACM)
				{
					if (vertices)
						delete vertices;	
					if (trilist)
						delete trilist;	
				}
			#endif

				if(useRACM)
				{
					#ifdef _USE_RACM
					if (pRACM)
						delete pRACM;
					#endif
				}
			for (MaterialList::iterator m = materiallist.begin(); m != materiallist.end(); m++)
				delete *m;
			materiallist.clear();			
			
			#ifdef _USE_OOC
			#ifdef _USE_OOC_DIRECTMM
			if (tree)
				deallocateFullMemoryMap(tree);
			if (indexlist)
				deallocateFullMemoryMap(indexlist);
			#else
			// deallocate node array
			if (tree)
				delete tree;
			// deallocate tri index array for leaves
			if (indexlist)
				delete indexlist;
			#endif
			#else
			// deallocate node array
			if (tree)
				delete [] tree;
			// deallocate tri index array for leaves
			if (indexlist)
				delete [] indexlist;
			#endif

			if (g_QuanErrs)
				delete g_QuanErrs;
			
			if (modelFileName)
				delete modelFileName;

			if (transformMatList)
				delete [] transformMatList;
			if (invTransformMatList)
				delete [] invTransformMatList;
		}
	}

	void calculateAABoundingBox() {
		if (!boundingBoxGenerated) {
		
			bb[0] = Vector3(FLT_MAX,FLT_MAX,FLT_MAX);
			bb[1] = Vector3(-FLT_MAX,-FLT_MAX,-FLT_MAX);

			if(!useRACM)
			{
				for (unsigned int i = 0; i < nTris; i++) {
					const Triangle &t = MODEL_GETTRI(i);
					for (int j = 0; j<3; j++) {
						const Vector3 &v = MODEL_GETVERTEX(t.p[j]);
						for (int k = 0; k<3; k++) {				
							if (v.e[k] < bb[0][k])
								bb[0][k] = v.e[k];
							if (v.e[k] > bb[1][k])
								bb[1][k] = v.e[k];
						}
					}	
				}
			}
			if(useRACM)
			{
				#ifdef _USE_RACM
				for (unsigned int i = 0; i < nTris; i++) {
					const COutTriangle &t = pRACM->GetTri(i);
					for (int j = 0; j<3; j++) {
						const CGeomVertex &v = pRACM->GetVertex((int)t.m_c[j].m_v);
						for (int k = 0; k<3; k++) {				
							if (v[k] < bb[0][k])
								bb[0][k] = v[k];
							if (v[k] > bb[1][k])
								bb[1][k] = v[k];
						}
					}	
				}
				#endif
			}

			treeStats.min = bb[0];
			treeStats.max = bb[1];

			bb[0] += translate_world;
			bb[1] += translate_world;

			boundingBoxGenerated = true;

			// generate fake texture params:
			Vector3 dim = bb[1] - bb[0];			
			i1 = dim.indexOfMaxComponent();
			i2 = (dim[(i1+1)%3] > dim[(i1+2)%3])?((i1+1)%3):((i1+2)%3);

			coordToU = tiling / dim[i1];
			coordToV = tiling / dim[i2];
		}		
	}

	void printModel() {
		LogManager *log = LogManager::getSingletonPtr();
		char output[500];

		if (isInstanced)
			sprintf(output, "Model:\t%s (instanced)", modelFileName);
		else
			sprintf(output, "Model:\t%s", modelFileName);
		log->logMessage(LOG_INFO, output);

		sprintf(output, "Tris:\t%u\tVertices:\t%u", nTris, nVerts);
		log->logMessage(LOG_INFO, output);
	}

	bool loadTreeFromFiles(const char* filename);

	// only the OOC version of load/save is available in OoC mode
	#if !defined(_USE_OOC) || defined(_USE_OOC_DIRECTMM)
	bool loadTreeFromFile(const char* filename);
	bool saveTreeToFile(const char* filename);
	#endif	

};

#endif