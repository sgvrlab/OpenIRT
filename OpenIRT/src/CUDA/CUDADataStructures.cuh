#ifndef GPU_DATA_STRUCTURES_H
#define GPU_DATA_STRUCTURES_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include "CUDAVector.cuh"
#include "CUDAMaterial.cuh"
#include "CommonOptions.h"

namespace CUDA
{

#define FLT_MAX 3.402823466e+38F
#define BSP_EPSILON 0.001f
#define INTERSECT_EPSILON 0.01f
#define TRI_INTERSECT_EPSILON 0.0001f

typedef cuVec3 Vector3;
typedef cuVec2 Vector2;

typedef union BVHNode_t {
	struct {
		unsigned int left;
		unsigned int right;
		Vector3 min;
		Vector3 max;
	};

	struct { // GPU
		uint4 low;
		uint4 high;
	};

} BVHNode;

typedef union Triangle_t {
	struct
	{
		unsigned int p[3];		// vertex indices
		unsigned char  i1,i2;	// planes to be projected to
		unsigned short material;	// Index of material in list
		Vector3 n;			    // normal vector (normalized)
		float d;				// d from plane equation
	};

	struct	// GPU
	{
		uint4 low;
		uint4 high;
	};

} Triangle;

typedef union Vertex_t {
	struct
	{
		Vector3 v;				// vertex geometry
		float dummy1;
		Vector3 n;				// normal vector
		float dummy2;
		Vector3 c;				// color
		float dummy3;
		Vector2 uv;				// Texture coordinate
	};

	struct	// GPU
	{
		uint4 quad0;
		uint4 quad1;
		uint4 quad2;
		uint4 quad3;
	};
} Vertex;

typedef struct AABB_t {
	struct
	{
		Vector3 min;
		Vector3 max;
	};

	__device__ AABB_t() {}
	__device__ AABB_t(const Vector3 &min, const Vector3 &max)
	{
		this->min = min;
		this->max = max;
	}
} AABB;

typedef union Photon_t {
	enum SplitChoice
	{
		SPLIT_CHOICE_ROUND_ROBIN,
		SPLIT_CHOICE_HIGHEST_VARIANCE,
		SPLIT_CHOICE_LONGEST_DIM
	};

	enum SplitAxis
	{
		SPLIT_AXIS_X,
		SPLIT_AXIS_Y,
		SPLIT_AXIS_Z,
		SPLIT_AXIS_LEAF,
		SPLIT_AXIS_NULL
	};

	struct {
		Vector3 power;
		unsigned short axis;
		unsigned char theta, phi;
		Vector3 pos;
		float dummy2;
	};

	struct {
		float4 low;
		float4 high;
	};

	__device__ void inline setDirection(const Vector3 &dir)
	{
		theta = (unsigned char)(acosf(dir.z) * 256.0f / PI);
		float temp = atan2f(dir.y, dir.x);
		// change range: (-pi, pi] -> [0, 2pi)
		temp = temp < 0 ? -temp : temp;
		phi = (unsigned char)(temp * 256.0f / (2*PI));
	}
} Photon;


typedef union PhotonVoxel_t {
	struct {
		Vector3 power;
		float dummy;
	};

	struct {
		float4 data;
	};
} PhotonVoxel;

typedef struct OctreeHeader_t
{
	struct {
		int dim;
		int maxDepth;
		Vector3 min;
		Vector3 max;
	};
} OctreeHeader;

typedef union Voxel_t {
	struct
	{
		unsigned int mat_Kd;
		unsigned int mat_Ks;
		unsigned short mat_d;
		unsigned short mat_Ns;
		/*
		int childIndex;
		Vector3 norm;
		float d;
		*/
		int childIndex;
		//Vector3 norm;
		unsigned char theta, phi;
		unsigned char geomBitmap[8];
		unsigned short m;
		float d;
	};
	/*
	struct
	{
		Vector3 col;
		int childIndex;
		Vector3 norm;
		float d;
	};
	*/

	struct	// GPU
	{
		float4 low;
		float4 high;
	};

	__device__ bool inline hasChild() {return (childIndex >> 2) != 0;}
	__device__ __host__ bool inline isLeaf() {return (childIndex & 0x1) == 0x1;}
	//__device__ bool inline isEmpty() {return !hasChild() && !isLeaf();}
	__device__ __host__ bool inline isEmpty() {return childIndex == 0;}
	__device__ bool inline hasLink2Low() const {return (childIndex & 0x2) == 0x2;}

	__device__ __host__ int inline getChildIndex() {return childIndex >> 2;}
	__device__ __host__ void inline setChildIndex(int index) {childIndex = (index == 0) ? 0x2 : (index << 2);}
//	__device__ __host__ void inline setChildIndex(int index) {childIndex = (index << 2) | 0x2;}
	__device__ __host__ void inline setLeaf() {childIndex |= 0x1;}

	__device__ Vector3 inline getKd() const {return Vector3((mat_Kd & 0xFF) / 255.0f, ((mat_Kd >> 8) & 0xFF) / 255.0f, ((mat_Kd >> 16) & 0xFF) / 255.0f);}
	__device__ Vector3 inline getKs() const {return Vector3((mat_Ks & 0xFF) / 255.0f, ((mat_Ks >> 8) & 0xFF) / 255.0f, ((mat_Ks >> 16) & 0xFF) / 255.0f);}
	__device__ float inline getD() const {return mat_d / 65535.0f;}
	__device__ float inline getNs() const {return (float)mat_Ns;}
	//__device__ __host__ int inline getLink2Low() {return *((int*)geomBitmap);}
	__device__ __host__ int inline getLink2Low() {return (geomBitmap[3] << 24) | (geomBitmap[2] << 16) | (geomBitmap[1] << 8) | geomBitmap[0];}
} Voxel;

typedef struct OOCVoxel_t
{
	struct
	{
		int rootChildIndex;
		int startDepth;
		int offset;
		int numVoxels;
		AABB rootBB;
	};
} OOCVoxel;


typedef struct Matrix_t {
	float x[16];

	__device__ __host__ void inline transformPosition(Vector3 &pos) const
	{
		float temp = pos.x * x[12] + pos.y * x[13] + pos.z * x[14] + x[15];
		Vector3 newPos(
			(pos.x * x[0] + pos.y * x[1] + pos.z * x[2] + x[3])/temp,
			(pos.x * x[4] + pos.y * x[5] + pos.z * x[6] + x[7])/temp,
			(pos.x * x[8] + pos.y * x[9] + pos.z * x[10] + x[11])/temp);
		pos = newPos;
	}

	__device__ __host__ void inline transformVector(Vector3 &vec) const
	{
		Vector3 newVec(
			vec.x * x[0] + vec.y * x[1] + vec.z * x[2],
			vec.x * x[4] + vec.y * x[5] + vec.z * x[6],
			vec.x * x[8] + vec.y * x[9] + vec.z * x[10]);
		vec = newVec;
	}

	__device__ __host__ Vector3 inline operator*(const Vector3 &vec) const
	{
		return Vector3(
			vec.x * x[0] + vec.y * x[1] + vec.z * x[2],
			vec.x * x[4] + vec.y * x[5] + vec.z * x[6],
			vec.x * x[8] + vec.y * x[9] + vec.z * x[10]);
	}

	__device__ __host__ Vector3 inline Tmul(const Vector3 &vec) const
	{
		return Vector3(
			vec.x * x[0] + vec.y * x[4] + vec.z * x[8],
			vec.x * x[1] + vec.y * x[5] + vec.z * x[9],
			vec.x * x[2] + vec.y * x[6] + vec.z * x[10]);
	}

	__device__ __host__ float4 inline operator*(const float4 &vec) const
	{
		return make_float4(
			vec.x * x[0] + vec.y * x[1] + vec.z * x[2] + vec.w * x[3],
			vec.x * x[4] + vec.y * x[5] + vec.z * x[6] + vec.w * x[7],
			vec.x * x[8] + vec.y * x[9] + vec.z * x[10] + vec.w * x[11],
			vec.x * x[12] + vec.y * x[13] + vec.z * x[14] + vec.w * x[15]
			);
	}

	__device__ __host__ float4 inline Tmul(const float4 &vec) const
	{
		return make_float4(
			vec.x * x[0] + vec.y * x[4] + vec.z * x[8] + vec.w * x[12],
			vec.x * x[1] + vec.y * x[5] + vec.z * x[9] + vec.w * x[13],
			vec.x * x[2] + vec.y * x[6] + vec.z * x[10] + vec.w * x[14],
			vec.x * x[3] + vec.y * x[7] + vec.z * x[11] + vec.w * x[15]
			);
	}

	__device__ __host__ Matrix_t inline operator*(const Matrix_t &mat) const
	{
		Matrix_t ret;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				float subt = 0.0;
				for (int k = 0; k < 4; k++)
					subt += x[i*4+k] * mat.x[k*4+j];
				ret.x[i*4+j] = subt;
			}
		}

		return ret;
	}

	__device__ __host__ Matrix_t inline transpose()
	{
		Matrix_t ret;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				ret.x[i*4+j] = x[j*4+i];
			}
		}

		return ret;
	}

	__device__ __host__ inline float det3 (float a, float b, float c, 
					   float d, float e, float f, 
					   float g, float h, float i)
	{ return a*e*i + d*h*c + g*b*f - g*e*c - d*b*i - a*h*f; }

	__device__ __host__ float determinant()
	{
		float det;
		det  = x[0*4+0] * det3(x[1*4+1], x[1*4+2], x[1*4+3], 
			x[2*4+1], x[2*4+2], x[2*4+3], 
			x[3*4+1], x[3*4+2], x[3*4+3]);
		det -= x[0*4+1] * det3(x[1*4+0], x[1*4+2], x[1*4+3],
			x[2*4+0], x[2*4+2], x[2*4+3],
			x[3*4+0], x[3*4+2], x[3*4+3]);
		det += x[0*4+2] * det3(x[1*4+0], x[1*4+1], x[1*4+3],
			x[2*4+0], x[2*4+1], x[2*4+3],
			x[3*4+0], x[3*4+1], x[3*4+3]);
		det -= x[0*4+3] * det3(x[1*4+0], x[1*4+1], x[1*4+2],
			x[2*4+0], x[2*4+1], x[2*4+2],
			x[3*4+0], x[3*4+1], x[3*4+2]);
		return det;
	}

	__device__ __host__ Matrix_t inverse()
	{
		float det = determinant();
		Matrix inverse;
		inverse.x[0*4+0]  = det3(x[1*4+1], x[1*4+2], x[1*4+3],
			x[2*4+1], x[2*4+2], x[2*4+3],
			x[3*4+1], x[3*4+2], x[3*4+3]) / det;
		inverse.x[0*4+1] = -det3(x[0*4+1], x[0*4+2], x[0*4+3],
			x[2*4+1], x[2*4+2], x[2*4+3],
			x[3*4+1], x[3*4+2], x[3*4+3]) / det;
		inverse.x[0*4+2]  = det3(x[0*4+1], x[0*4+2], x[0*4+3],
			x[1*4+1], x[1*4+2], x[1*4+3],
			x[3*4+1], x[3*4+2], x[3*4+3]) / det;
		inverse.x[0*4+3] = -det3(x[0*4+1], x[0*4+2], x[0*4+3],
			x[1*4+1], x[1*4+2], x[1*4+3],
			x[2*4+1], x[2*4+2], x[2*4+3]) / det;

		inverse.x[1*4+0] = -det3(x[1*4+0], x[1*4+2], x[1*4+3],
			x[2*4+0], x[2*4+2], x[2*4+3],
			x[3*4+0], x[3*4+2], x[3*4+3]) / det;
		inverse.x[1*4+1]  = det3(x[0*4+0], x[0*4+2], x[0*4+3],
			x[2*4+0], x[2*4+2], x[2*4+3],
			x[3*4+0], x[3*4+2], x[3*4+3]) / det;
		inverse.x[1*4+2] = -det3(x[0*4+0], x[0*4+2], x[0*4+3],
			x[1*4+0], x[1*4+2], x[1*4+3],
			x[3*4+0], x[3*4+2], x[3*4+3]) / det;
		inverse.x[1*4+3]  = det3(x[0*4+0], x[0*4+2], x[0*4+3],
			x[1*4+0], x[1*4+2], x[1*4+3],
			x[2*4+0], x[2*4+2], x[2*4+3]) / det;

		inverse.x[2*4+0]  = det3(x[1*4+0], x[1*4+1], x[1*4+3],
			x[2*4+0], x[2*4+1], x[2*4+3],
			x[3*4+0], x[3*4+1], x[3*4+3]) / det;
		inverse.x[2*4+1] = -det3(x[0*4+0], x[0*4+1], x[0*4+3],
			x[2*4+0], x[2*4+1], x[2*4+3],
			x[3*4+0], x[3*4+1], x[3*4+3]) / det;
		inverse.x[2*4+2]  = det3(x[0*4+0], x[0*4+1], x[0*4+3],
			x[1*4+0], x[1*4+1], x[1*4+3],
			x[3*4+0], x[3*4+1], x[3*4+3]) / det;
		inverse.x[2*4+3] = -det3(x[0*4+0], x[0*4+1], x[0*4+3],
			x[1*4+0], x[1*4+1], x[1*4+3],
			x[2*4+0], x[2*4+1], x[2*4+3]) / det;


		inverse.x[3*4+0] = -det3(x[1*4+0], x[1*4+1], x[1*4+2],
			x[2*4+0], x[2*4+1], x[2*4+2],
			x[3*4+0], x[3*4+1], x[3*4+2]) / det;
		inverse.x[3*4+1] =  det3(x[0*4+0], x[0*4+1], x[0*4+2],
			x[2*4+0], x[2*4+1], x[2*4+2],
			x[3*4+0], x[3*4+1], x[3*4+2]) / det;
		inverse.x[3*4+2] = -det3(x[0*4+0], x[0*4+1], x[0*4+2],
			x[1*4+0], x[1*4+1], x[1*4+2],
			x[3*4+0], x[3*4+1], x[3*4+2]) / det;
		inverse.x[3*4+3] =  det3(x[0*4+0], x[0*4+1], x[0*4+2],
			x[1*4+0], x[1*4+1], x[1*4+2],
			x[2*4+0], x[2*4+1], x[2*4+2]) / det;   

		return inverse;
	}

} Matrix;

typedef struct HitPoint_t {
	float t;				// Param of ray
	unsigned int model;
	unsigned int material;	// Rerefence to material
	Vector3 n;
	Vector2 uv;				// tex coords
	Vector3 x;
} HitPoint;

typedef struct Ray_t {
	Vector3 ori;
	Vector3 dir;
	Vector3 invDir;
	char posNeg[4];

	__device__ void inline set(const Vector3 &o, const Vector3 &d)
	{
		ori = o;
		dir = d.normalize();
		invDir.x = 1.0f / dir.x;
		invDir.y = 1.0f / dir.y;
		invDir.z = 1.0f / dir.z;

		posNeg[0] =  (dir.x >= 0 ? 1 : 0);
		posNeg[1] =  (dir.y >= 0 ? 1 : 0);
		posNeg[2] =  (dir.z >= 0 ? 1 : 0);
		posNeg[3] =  (posNeg[0] << 2) | (posNeg[1] << 1) | posNeg[2];
	}

	__device__ bool inline BoxIntersect(const Vector3 &minBB, const Vector3 &maxBB, float &t0, float &t1) const
	{
		t0 = ((posNeg[0] ? minBB.x : maxBB.x) - ori.x) * invDir.x;
		t1 = ((posNeg[0] ? maxBB.x : minBB.x) - ori.x) * invDir.x;

		t0 = fmaxf(((posNeg[1] ? minBB.y : maxBB.y) - ori.y) * invDir.y, t0);
		t1 = fminf(((posNeg[1] ? maxBB.y : minBB.y) - ori.y) * invDir.y, t1);

		t0 = fmaxf(((posNeg[2] ? minBB.z : maxBB.z) - ori.z) * invDir.z, t0);
		t1 = fminf(((posNeg[2] ? maxBB.z : minBB.z) - ori.z) * invDir.z, t1);

		return (t0 <= t1);
	}

	__device__ bool inline BoxIntersect(const float minx, const float miny, const float minz, const float maxx, const float maxy, const float maxz, float &t0, float &t1) const
	{
		t0 = ((posNeg[0] ? minx : maxx) - ori.x) * invDir.x;
		t1 = ((posNeg[0] ? maxx : minx) - ori.x) * invDir.x;

		t0 = fmaxf(((posNeg[1] ? miny : maxy) - ori.y) * invDir.y, t0);
		t1 = fminf(((posNeg[1] ? maxy : miny) - ori.y) * invDir.y, t1);

		t0 = fmaxf(((posNeg[2] ? minz : maxz) - ori.z) * invDir.z, t0);
		t1 = fminf(((posNeg[2] ? maxz : minz) - ori.z) * invDir.z, t1);

		return (t0 <= t1);
	}

	__device__ void inline transform(const Matrix &matrix)
	{
		/*
		float temp = ori.x * matrix[12] + ori.y * matrix[13] + ori.z * matrix[14] + matrix[15];
		Vector3 newOri(
			(ori.x * matrix[0] + ori.y * matrix[1] + ori.z * matrix[2] + matrix[3])/temp,
			(ori.x * matrix[4] + ori.y * matrix[5] + ori.z * matrix[6] + matrix[7])/temp,
			(ori.x * matrix[8] + ori.y * matrix[9] + ori.z * matrix[10] + matrix[11])/temp);
		Vector3 newDir(
			dir.x * matrix[0] + dir.y * matrix[1] + dir.z * matrix[2],
			dir.x * matrix[4] + dir.y * matrix[5] + dir.z * matrix[6],
			dir.x * matrix[8] + dir.y * matrix[9] + dir.z * matrix[10]);
		ori = newOri;
		dir = newDir;
		*/
		matrix.transformPosition(ori);
		matrix.transformVector(dir);
		
		invDir.x = 1.0f / dir.x;
		invDir.y = 1.0f / dir.y;
		invDir.z = 1.0f / dir.z;

		posNeg[0] =  (dir.x >= 0 ? 1 : 0);
		posNeg[1] =  (dir.y >= 0 ? 1 : 0);
		posNeg[2] =  (dir.z >= 0 ? 1 : 0);
		posNeg[3] =  (posNeg[0] << 2) | (posNeg[1] << 1) | posNeg[2];
	}

} Ray;

typedef struct ExtraRayInfo_t
{
	float pixelX;
	float pixelY;
	//unsigned int frame2;
	unsigned int seed;
	unsigned char mask;

	__device__ __host__ bool hasHit() {return (mask & 0x1) != 0;}
	__device__ __host__ bool isSpecular() {return (mask & 0x2) != 0;}
	__device__ __host__ bool wasBounced() {return (mask & 0x4) != 0;}

	__device__ __host__ void setHasHit(bool val) {mask = val ? (mask | 0x1) : (mask & ~0x1);}
	__device__ __host__ void setIsSpecular(bool val) {mask = val ? (mask | 0x2) : (mask & ~0x2);}
	__device__ __host__ void setWasBounced(bool val) {mask = val ? (mask | 0x4) : (mask & ~0x4);}

	__device__ __host__ ExtraRayInfo_t() {mask = 0;}
	//__device__ __host__ void setIsSpecular(bool val) {mask |= 0x2;}
	//__device__ __host__ void setWasBounced(bool val) {mask |= 0x4;}

	//__device__ __host__ void resetHasHit() {mask &= ~0x1;}
	//__device__ __host__ void resetIsSpecular() {mask &= ~0x2;}
	//__device__ __host__ void resetWasBounced() {mask &= ~0x4;}
	//int hasHit;
	//bool isSpecular;
	//bool wasBounced;
} ExtraRayInfo;

typedef struct HColor_t
{
	struct {
	Vector3 color1;
	Vector3 color2;
	float numIntergration;
	float numIteration;
	};
	//HColor_t(int numIntergration = 0) : numIntergration(numIntergration) {}
} HColor;

typedef struct Camera_t {
	struct {
	// camera name
	char name[256];

	// camera position and orientation
	Vector3 eye;
	Vector3 center;
	Vector3 up;

	// projection properties
	float fovy;
	float aspect;
	float zNear;
	float zFar;

	// viewing transformation vectors (normalized)
	Vector3 vLookAt;
	Vector3 vRight;
	Vector3 vUp;

	// for efficiency
	Vector3 scaledRight;
	Vector3 scaledUp;

	Vector3 corner;	// left top corner of image plane on world coordinate space
	};

	__device__ inline void getRayWithOrigin(Ray &ray, float a, float b) 
	{
		Vector3 target = corner + a*scaledRight + b*scaledUp;

		ray.ori = eye;
		ray.dir = target-eye;
		ray.dir = ray.dir.normalize();

		ray.invDir.x = 1.0f / ray.dir.x;
		ray.invDir.y = 1.0f / ray.dir.y;
		ray.invDir.z = 1.0f / ray.dir.z;

		ray.posNeg[0] =  (ray.dir.x >= 0 ? 1 : 0);
		ray.posNeg[1] =  (ray.dir.y >= 0 ? 1 : 0);
		ray.posNeg[2] =  (ray.dir.z >= 0 ? 1 : 0);
		ray.posNeg[3] =  (ray.posNeg[0] << 2) | (ray.posNeg[1] << 1) | ray.posNeg[2];
	}

	__device__ inline void getRayWithOrigin(Vector3 &ori, Vector3 &dir, float a, float b) 
	{
		Vector3 target = corner + a*scaledRight + b*scaledUp;

		ori = eye;
		dir = target-eye;
		dir = dir.normalize();
	}

	__host__ bool operator == (const Camera_t &camera)
	{
		if(eye != camera.eye) return false;
		if(center != camera.center) return false;
		if(up != camera.up) return false;
		if(fovy != camera.fovy) return false;
		if(aspect != camera.aspect) return false;
		if(zNear != camera.zNear) return false;
		if(zFar != camera.zFar) return false;
		return true;
	}

	__host__ bool operator != (const Camera_t &camera)
	{
		return !(*this == camera);
	}
} Camera;


typedef struct Model_t {
	struct {
		int numVerts;
		int numTris;
		int numNodes;
		int numMats;

		Vertex *verts;
		Triangle *tris;
		BVHNode *nodes;
		Material *mats;

		float transfMatrix[16];
		float invTransfMatrix[16];
	};
} Model;

typedef struct SceneNode_t {
	struct {
		int modelIdx;
		int numChilds;
		Vector3 bbMin;
		Vector3 bbMax;
		int childIdx[MAX_NUM_MODELS];
	};
} SceneNode;

typedef struct Image_t {
	int width;
	int height;
	int bpp;
	float *data;
} Image;

typedef struct Scene_t {
	struct {
		int numModels;
		Model *models;

		int numEmitters;
		Emitter *emitters;

		Image envMap[6];
		Vector3 envColor;

		Vector3 bbMin;
		Vector3 bbMax;

		SceneNode sceneGraph[MAX_NUM_MODELS*2];
	};
} Scene;


typedef struct SampleData_t {
	struct
	{
		// for shading
		Vector3 color1;
		Vector3 color2;
		float numIntergration;
		float numIteration;

		// for gathering rays
		int isHit;
		Vector3 hitPoint;
		Vector3 hitNormal;
		MaterialSampler hitMat;
		Vector3 inDir;
		Vector3 brdf;

		// for filtering
		int hasHit;
		Vector3 summedHitPoint;
		Vector3 summedHitNormal;

		// for ambient occlusion
		int numAOSample;
		int numAOHit;
	};

	__device__ Vector3 getFinalColor()
	{
		Vector3 color(0.0f);

		if(numIntergration > 0)
		{
			color.e[0] = color1.e[0] / numIntergration;
			color.e[1] = color1.e[1] / numIntergration;
			color.e[2] = color1.e[2] / numIntergration;
		}
		else
		{
			color.e[0] = color1.e[0];
			color.e[1] = color1.e[1];
			color.e[2] = color1.e[2];
		}

		if(numIteration > 0)
		{
			color.e[0] += color2.e[0] / numIteration;
			color.e[1] += color2.e[1] / numIteration;
			color.e[2] += color2.e[2] / numIteration;
		}

		color.e[0] = fmaxf(color.e[0], 0.0f);
		color.e[1] = fmaxf(color.e[1], 0.0f);
		color.e[2] = fmaxf(color.e[2], 0.0f);

		color.e[0] = fminf(color.e[0], 1.0f);
		color.e[1] = fminf(color.e[1], 1.0f);
		color.e[2] = fminf(color.e[2], 1.0f);
		return color;
	}

	__device__ Vector3 getFinalColor(int mask)
	{
		Vector3 color(0.0f);

		if((mask & 0x1) != 0)
		{
			if(numIntergration > 0)
			{
				color.e[0] = color1.e[0] / numIntergration;
				color.e[1] = color1.e[1] / numIntergration;
				color.e[2] = color1.e[2] / numIntergration;
			}
			else
			{
				color.e[0] = color1.e[0];
				color.e[1] = color1.e[1];
				color.e[2] = color1.e[2];
			}
		}

		if((mask & 0x2) != 0)
		{
			if(numIteration > 0)
			{
				color.e[0] += color2.e[0] / numIteration;
				color.e[1] += color2.e[1] / numIteration;
				color.e[2] += color2.e[2] / numIteration;
			}
		}

		color.e[0] = fmaxf(color.e[0], 0.0f);
		color.e[1] = fmaxf(color.e[1], 0.0f);
		color.e[2] = fmaxf(color.e[2], 0.0f);

		color.e[0] = fminf(color.e[0], 1.0f);
		color.e[1] = fminf(color.e[1], 1.0f);
		color.e[2] = fminf(color.e[2], 1.0f);
		return color;
	}

	__host__ __device__ void reset()
	{
		/*
		int count = sizeof(SampleData_t);
		for(int i=0;i<count;i++)
		{
			*(((unsigned char*)this) + i) = 0;
		}
		*/
		isHit = 0;
		hitPoint.reset();
		hitNormal.reset();
		hitMat.reset();
		inDir.reset();
		color1.reset();
		color2.reset();
		brdf.reset();
		summedHitPoint.reset();
		summedHitNormal.reset();
		numIntergration = 0;
		numIteration = 0;
		numAOSample = 0;
		numAOHit = 0;
	}

} SampleData;

typedef struct Controller_t
{
	enum FilterType
	{
		NONE,
		BOX,
		G_BUFFER
	};

	enum TileOrderingType
	{
		RANDOM,
		ROW_BY_ROW,
		Z_CURVE,
		HIGHEST_SALIENT_TILE_FIRST,
		CURSOR_GUIDED
	};

	bool useZCurveOrdering;
	bool shadeLocalIllumination;
	bool useShadowRays;
	bool gatherPhotons;
	bool showLights;
	bool useAmbientOcclusion;
	bool printLog;
	bool drawBackground;
	int pathLength;
	int numShadowRays;
	int numGatheringRays;
	int threadBlockSize;
	float timeLimit;
	int tileSize;
	FilterType filterType;
	int filterWindowSize;
	int filterIteration;
	float filterParam1;
	float filterParam2;
	float filterParam3;
	TileOrderingType tileOrderingType;
	int sizeMBForOOCVoxel;
	int warpSizeS;
	int warpSizeG;
	float AODistance;
	float envMapWeight;
	float envColWeight;
} Controller;

typedef struct StatData_t {
	int numSamples;
	double sumSamples;
	double sumSamples2;
} StatData;

typedef struct SSDODataEnvMap_t
{

	struct {
		Texture envMap[6];	// cube map
		Texture PEM;
		Vector3 envColor;
	};
} SSDODataEnvMap;


typedef struct SSDODataRuntime_t
{

	struct {
		int width;
		int height;
		unsigned char *output;
		int numResources;
		cudaGraphicsResource_t resourceList[MAX_NUM_OPENGL_RESOURCES];
		Camera camera;
		Matrix modelViewMatrix;
		Matrix modelViewProjectionMatrix;
		Matrix invModelViewMatrix;
		Matrix invModelViewProjectionMatrix;
	};
} SSDODataRuntime;

typedef struct SSDODataLight_t
{

	struct {
		bool soft;
		int width;
		int height;
		int numResources;
		cudaGraphicsResource_t resourceList[MAX_NUM_OPENGL_RESOURCES];
		int numEmitters;
		Emitter emitterList[MAX_NUM_EMITTERS];
		Matrix lightViewProjectionMatrix[MAX_NUM_EMITTERS];
		float lightProjectZNear[MAX_NUM_EMITTERS];
		float lightProjectZFar[MAX_NUM_EMITTERS];
	};
} SSDODataLight;

}

#endif // #ifndef DATA_TYPE_GPU_H
