#ifndef DATA_TYPE_GPU_H
#define DATA_TYPE_GPU_H

typedef union BVHNode_t {
	struct {
		unsigned int left;
		unsigned int right;
		cuVec3 min;
		cuVec3 max;
	};

	struct { // GPU
		uint4 low;
		uint4 high;
	};

} BVHNode, *BVHNodePtr;

typedef union Triangle_t {
	struct
	{
		unsigned int p[3];		// vertex indices
		cuVec3 n;			    // normal vector (normalized)
		float d;				// d from plane equation
		unsigned char  i1,i2;	// planes to be projected to
		unsigned short material;	// Index of material in list
	};

	struct	// GPU
	{
		uint4 low;
		uint4 high;
	};

} Triangle, *TrianglePtr;

typedef union Vertex_t {
	struct
	{
		cuVec3 v;				// vertex geometry
		cuVec3 n;				// normal vector
		cuVec3 c;				// color
		cuVec2 uv;				// Texture coordinate
		unsigned char dummy[20];
	};

	struct	// GPU
	{
		uint4 quad1;
		uint4 quad2;
		uint4 quad3;
		uint4 quad4;
	};
} Vertex, *VertexPtr;

typedef struct HitPoint_t {
	float t;		 // Param of ray
	unsigned int m; // Rerefence to material
	cuVec3 n;
	cuVec2 uv;		// tex coords
} HitPoint, *HitPointPtr;

typedef struct Ray_t {
	cuVec3 ori;
	cuVec3 dir;
	cuVec3 invDir;
	char posNeg[4];

	__device__ bool CUDA_INLINE BoxIntersect(const float minx, const float miny, const float minz, const float maxx, const float maxy, const float maxz, float &t0, float &t1)
	{
		/*
		t0 = (box[posNeg2[0]].x - ori.x) * invDir.x;
		t1 = (box[posNeg1[0]].x - ori.x) * invDir.x;

		t0 = fmaxf((box[posNeg2[1]].y - ori.y) * invDir.y, t0);
		t1 = fminf((box[posNeg1[1]].y - ori.y) * invDir.y, t1);

		t0 = fmaxf((box[posNeg2[2]].z - ori.z) * invDir.z, t0);
		t1 = fminf((box[posNeg1[2]].z - ori.z) * invDir.z, t1);;

		//printf("%f %f %f, %f %f %f\n", box[0].x, box[0].y, box[0].z, box[1].x, box[1].y, box[1].z);
		return (t0 <= t1);
		*/
		t0 = ((posNeg[0] ? minx : maxx) - ori.x) * invDir.x;
		t1 = ((posNeg[0] ? maxx : minx) - ori.x) * invDir.x;

		t0 = fmaxf(((posNeg[1] ? miny : maxy) - ori.y) * invDir.y, t0);
		t1 = fminf(((posNeg[1] ? maxy : miny) - ori.y) * invDir.y, t1);

		t0 = fmaxf(((posNeg[2] ? minz : maxz) - ori.z) * invDir.z, t0);
		t1 = fminf(((posNeg[2] ? maxz : minz) - ori.z) * invDir.z, t1);

		//printf("t0 = %f, t1 = %f\n", t0, t1);
		//printf("t0 = %f, t1 = %f\n%f %f %f\n%f %f %f\n", t0, t1, *minx, *miny, *minz, *maxx, *maxy, *maxz);
		return (t0 <= t1);
	}

	/*
	__device__ cuVec2 CUDA_INLINE getOriUV(const int &i1, const int &i2)
	{
		return (i1 == 1) ? cuVec2(ori.y, ori.z) : ((i2 == 1) ? cuVec2(ori.x, ori.y) : cuVec2(ori.x, ori.z));
	}

	__device__ cuVec2 CUDA_INLINE getDirUV(const int &i1, const int &i2)
	{
		return (i1 == 1) ? cuVec2(dir.y, dir.z) : ((i2 == 1) ? cuVec2(dir.x, dir.y) : cuVec2(dir.x, dir.z));
	}
	*/
} Ray, *RayPtr;

typedef struct Camera_t {
	struct {
		cuVec3 center;
		cuVec3 corner;
		cuVec3 across;
		cuVec3 up;
		float fovy;
	};
	__device__ CUDA_INLINE void getRayWithOrigin(Ray &ray, float a, float b) 
	{
		cuVec3 target = corner + across*a + up*b;
		ray.ori = center;
		ray.dir = target-center;
		ray.dir = ray.dir.normalize();

		ray.invDir.x = 1.0f / ray.dir.x;
		ray.invDir.y = 1.0f / ray.dir.y;
		ray.invDir.z = 1.0f / ray.dir.z;

		ray.posNeg[0] =  (ray.dir.x >= 0 ? 1 : 0);
		ray.posNeg[1] =  (ray.dir.y >= 0 ? 1 : 0);
		ray.posNeg[2] =  (ray.dir.z >= 0 ? 1 : 0);
	}

	CUDA_INLINE void getRayWithOriginHost(Ray &ray, float a, float b) 
	{
		cuVec3 target;
		target.x = corner.x + across.x*a + up.x*b;
		target.y = corner.y + across.y*a + up.y*b;
		target.z = corner.z + across.z*a + up.z*b;
		ray.ori = center;
		ray.dir.x = target.x-center.x;
		ray.dir.y = target.y-center.y;
		ray.dir.z = target.z-center.z;
		// normalize dir
		float lenDir = sqrtf(ray.dir.x*ray.dir.x + ray.dir.y*ray.dir.y + ray.dir.z*ray.dir.z);
		ray.dir.x = ray.dir.x / lenDir;
		ray.dir.y = ray.dir.y / lenDir;
		ray.dir.z = ray.dir.z / lenDir;

		ray.invDir.x = 1.0f / ray.dir.x;
		ray.invDir.y = 1.0f / ray.dir.y;
		ray.invDir.z = 1.0f / ray.dir.z;

		ray.posNeg[0] =  (ray.dir.x >= 0 ? 1 : 0);
		ray.posNeg[1] =  (ray.dir.y >= 0 ? 1 : 0);
		ray.posNeg[2] =  (ray.dir.z >= 0 ? 1 : 0);
	}
} Camera, *CameraPtr;

typedef struct Emitter_t {
	struct {
	cuVec3 emissionIntensity;		// emission separated into the three color bands
	cuVec3 emissionColor;			// emission separated into the three color bands, but limited to 0.0-1.0
	float summedIntensity;		// sum of all three emission components, used for choosing emitters
	float cutOffDistance;
	bool isViewerLight;
	cuVec3 p[3];				// the three vertices of the triangle
	cuVec3 n;					// the normal
	float area, areaInv;		// area of this emitter
	};

	// Samples a point on the light source given by the two random
	// variable u and v (in [0..1])
	__device__ void sample(const float u, const float v, cuVec3 &pointOnEmitter) {
		float temp = sqrt(1.0f - u);
		float beta = 1.0f - temp;
		float gamma = temp*v;

		pointOnEmitter = (1.0f - beta - gamma)*p[0] + beta*p[1] + gamma*p[2];
	}

	// Samples a direction from this light source given by the two
	// random variables u and v (in [0..1])
	__device__ void sampleDirection(const float u, const float v, cuVec3 &newDirection) {
		cuVec3 m1(1.0f, 0.0f, 0.0f);
		cuVec3 m2(0.0f, 1.0f, 0.0f);
		
		// Importance sample according to cosine
		float phi = 2.0f * 3.141592f * u;
		float r = sqrtf(v);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from normal
		cuVec3 U = n.cross(m1);
		if (U.length() < 0.01f)
			U = n.cross(m2); 
		cuVec3 V = n.cross(U);

		// use coordinates in basis:
		newDirection = x * U + y * V + z * n;
	}

} Emitter, *EmitterPtr;

typedef struct MaterialDiffuse_t {
	unsigned int type;
	cuVec3 rgb;
} MaterialDiffuse;

#endif // #ifndef DATA_TYPE_GPU_H
