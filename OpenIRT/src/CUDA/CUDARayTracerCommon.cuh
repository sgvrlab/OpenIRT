/********************************************************************
	created:	2011/08/03
	file path:	d:\Projects\Redering\OpenIRT\src\CUDA
	file base:	CUDARayTracerCommon
	file ext:	cuh
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Common variables and functions for ray tracing using CUDA
*********************************************************************/
/*
#include "cuPrintf.cu"
//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define CUPRINTF cuPrintf 
#else						//Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
								blockIdx.y*gridDim.x+blockIdx.x,\
								threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
								__VA_ARGS__)
#endif
*/
__constant__ Camera c_camera;
__constant__ Scene c_scene;
__constant__ Controller c_controller;
__constant__ Emitter c_emitterList[MAX_NUM_EMITTERS];
__constant__ int c_imgWidth;
__constant__ int c_imgHeight;
__constant__ int c_imgBpp;
__constant__ Vector3 c_envCol;
__constant__ bool c_hasEnvMap;

int h_frame;
Camera h_camera;
Scene h_scene;
Image h_image;
Controller h_controller;
unsigned char *d_cacheMem;
unsigned char *d_imageData;
float *d_summedImageData;
float *d_summedImageHitData;
float *d_summedImageDepthData;
float *d_summedImageNormalData;
float *d_envMapMem;

texture<float4, cudaTextureType1D, cudaReadModeElementType> t_model;
texture<float4, cudaTextureType1D, cudaReadModeElementType> t_envMap;

__constant__ size_t c_offsetModels[MAX_NUM_MODELS];
__constant__ size_t c_offsetVerts[MAX_NUM_MODELS];
__constant__ size_t c_offsetTris[MAX_NUM_MODELS];
__constant__ size_t c_offsetNodes[MAX_NUM_MODELS];
__constant__ size_t c_offsetMats[MAX_NUM_MODELS];

__constant__ Matrix c_transfMatrix[MAX_NUM_MODELS];
__constant__ Matrix c_invTransfMatrix[MAX_NUM_MODELS];

size_t h_offsetModels[MAX_NUM_MODELS];
size_t h_offsetVerts[MAX_NUM_MODELS];
size_t h_offsetTris[MAX_NUM_MODELS];
size_t h_offsetNodes[MAX_NUM_MODELS];
size_t h_offsetMats[MAX_NUM_MODELS];

float h_transfMatrix[MAX_NUM_MODELS][16];
float h_invTransfMatrix[MAX_NUM_MODELS][16];

size_t h_modelSize[MAX_NUM_MODELS];

#define RAY_OFFSET(x, y) ((x) + (y) * c_imgWidth)

#define VERT_FETCH(model, id, offset, ret) ((ret) = tex1Dfetch(t_model, c_offsetModels[model]+c_offsetVerts[model]+(id)*sizeof(Vertex)/16+(offset)))
#define TRI_FETCH(model, id, offset, ret) ((ret) = tex1Dfetch(t_model, c_offsetModels[model]+c_offsetTris[model]+(id)*sizeof(Triangle)/16+(offset)))
#define NODE_FETCH(model, id, offset, ret) ((ret) = tex1Dfetch(t_model, c_offsetModels[model]+c_offsetNodes[model]+(id)*sizeof(BVHNode)/16+(offset)))
#define MAT_FETCH(model, id, offset, ret) ((ret) = tex1Dfetch(t_model, c_offsetModels[model]+c_offsetMats[model]+(id)*sizeof(Material)/16+(offset)))
#define TEX_FETCH(model, texture, u, v, ret) ((ret) = tex1Dfetch(t_model, c_offsetModels[model]+(texture).getTexOffset(u, v)))

__device__ inline void shadeEnvironmentMap(const Vector3 &dir, float4 &color)
{
	// Order: Right, Left, Up, Down, Back, Front	
	int faceToU[] = { 2, 2, 0, 0, 0, 0 };
	int faceToV[] = { 1, 1, 2, 2, 1, 1 };
	float faceToUSign[] = { 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
	float faceToVSign[] = { 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f };

	int idx = dir.indexOfMaxAbsComponent();
	int face = (dir.e[idx] > 0.0)?0:1;
	face += 2*idx;

	int idx1 = faceToU[face];
	int idx2 = faceToV[face];

	float u = (faceToUSign[face]*dir.e[idx1] / fabsf(dir.e[idx]) + 1.0f) / 2.0f;
	float v = (faceToVSign[face]*dir.e[idx2] / fabsf(dir.e[idx]) + 1.0f) / 2.0f;

	const int &texWidth = c_scene.envMap[0].width;
	const int &texHeight = c_scene.envMap[0].height;
	const int texOffset = ((int)((0.999999f-v)*(float)texHeight) % texHeight) * texWidth + ((int)(u*(float)texWidth) % texWidth);
	color = tex1Dfetch(t_envMap, texWidth*texHeight*face + texOffset);
};

__device__ void getMaterial(Material &material, const HitPoint &hit)
{
	MAT_FETCH(hit.model, hit.material, 0, material._0);
	MAT_FETCH(hit.model, hit.material, 1, material._1);
	MAT_FETCH(hit.model, hit.material, 2, material._2);
	MAT_FETCH(hit.model, hit.material, 3, material._3);
	MAT_FETCH(hit.model, hit.material, 4, material._4);
	MAT_FETCH(hit.model, hit.material, 5, material._5);
	MAT_FETCH(hit.model, hit.material, 6, material._6);
	MAT_FETCH(hit.model, hit.material, 7, material._7);
	MAT_FETCH(hit.model, hit.material, 8, material._8);
	MAT_FETCH(hit.model, hit.material, 9, material._9);
	MAT_FETCH(hit.model, hit.material, 10, material._10);
	
	if(material.map_Ka.data)
	{
		float4 tex;
		TEX_FETCH(hit.model, material.map_Ka, hit.uv.x, hit.uv.y, tex);
		material.mat_Ka.x *= tex.x;
		material.mat_Ka.y *= tex.y;
		material.mat_Ka.z *= tex.z;
	}

	if(material.map_Kd.data)
	{
		float4 tex;
		TEX_FETCH(hit.model, material.map_Kd, hit.uv.x, hit.uv.y, tex);
		material.mat_Kd.x *= tex.x;
		material.mat_Kd.y *= tex.y;
		material.mat_Kd.z *= tex.z;

		material.mat_Ka.x *= tex.x;
		material.mat_Ka.y *= tex.y;
		material.mat_Ka.z *= tex.z;
	}
}

__device__ inline void getMaterial2(Material &material, float4 &resultKa, float4 &resultKd, float4 &resultKs, const HitPoint &hit)
{
	MAT_FETCH(hit.model, hit.material, 0, material._0);
	MAT_FETCH(hit.model, hit.material, 1, material._1);
	MAT_FETCH(hit.model, hit.material, 2, material._2);
	MAT_FETCH(hit.model, hit.material, 3, material._3);
	MAT_FETCH(hit.model, hit.material, 4, material._4);
	MAT_FETCH(hit.model, hit.material, 5, material._5);
	MAT_FETCH(hit.model, hit.material, 6, material._6);
	MAT_FETCH(hit.model, hit.material, 7, material._7);
	MAT_FETCH(hit.model, hit.material, 8, material._8);
	MAT_FETCH(hit.model, hit.material, 9, material._9);
	MAT_FETCH(hit.model, hit.material, 10, material._10);
	
	if(material.map_Ka.data)
	{
		float4 tex;
		TEX_FETCH(hit.model, material.map_Ka, hit.uv.x, hit.uv.y, tex);
		resultKa.x = material.mat_Ka.x * tex.x;
		resultKa.y = material.mat_Ka.y * tex.y;
		resultKa.z = material.mat_Ka.z * tex.z;
		resultKa.w = tex.w;

		resultKd.x = material.mat_Kd.x * tex.x;
		resultKd.y = material.mat_Kd.y * tex.y;
		resultKd.z = material.mat_Kd.z * tex.z;
		resultKd.w = tex.w;
	}

	if(material.map_Kd.data)
	{
		float4 tex;
		TEX_FETCH(hit.model, material.map_Kd, hit.uv.x, hit.uv.y, tex);
		resultKa.x = material.mat_Ka.x * tex.x;
		resultKa.y = material.mat_Ka.y * tex.y;
		resultKa.z = material.mat_Ka.z * tex.z;
		resultKa.w = tex.w;

		resultKd.x = material.mat_Kd.x * tex.x;
		resultKd.y = material.mat_Kd.y * tex.y;
		resultKd.z = material.mat_Kd.z * tex.z;
		resultKd.w = tex.w;
	}
}

/*
__device__ inline void getMaterial(Vector3 &matAmbient, Vector3 &matDiffuse, const HitPoint &hit)
{
	float4 mat0;
	float4 mat1;
	MAT_FETCH(hit.model, hit.material, 0, mat0);
	MAT_FETCH(hit.model, hit.material, 1, mat1);

	matAmbient.set(mat0.x, mat0.y, mat0.z);
	matDiffuse.set(mat0.w, mat1.x, mat1.y);

	Texture map_Ka, map_Kd;
	MAT_FETCH(hit.model, hit.material, 8, map_Ka.quad);
	MAT_FETCH(hit.model, hit.material, 9, map_Kd.quad);

	if(map_Ka.data)
	{
		TEX_FETCH(hit.model, map_Ka, hit.uv.x, hit.uv.y, mat0);
		matAmbient.x *= mat0.x;
		matAmbient.y *= mat0.y;
		matAmbient.z *= mat0.z;
	}

	if(map_Kd.data)
	{
		TEX_FETCH(hit.model, map_Kd, hit.uv.x, hit.uv.y, mat0);
		matDiffuse.x *= mat0.x;
		matDiffuse.y *= mat0.y;
		matDiffuse.z *= mat0.z;
	}
}
*/

__device__ bool RayTriIntersect(const Ray &ray, int model, unsigned int triID, HitPoint &hit, Material &hitMat, float tmax, float mag, unsigned int seed)
{
	float alpha, beta;
	//float u0, u1, u2;
	//float v0, v1, v2;
	unsigned int p0, p1, p2;

	float4 fetchedValue;
	TRI_FETCH(model, triID, 1, fetchedValue);
	float vdot = (fetchedValue.x*ray.dir.x + fetchedValue.y*ray.dir.y + fetchedValue.z*ray.dir.z);
	float vdot2 = (fetchedValue.x*ray.ori.x + fetchedValue.y*ray.ori.y + fetchedValue.z*ray.ori.z);

	if(!(vdot > -FLT_MAX && vdot < FLT_MAX)) return false;

	float t = (fetchedValue.w - vdot2) / vdot;

	// if either too near or further away than a previous hit, we stop
	if (t < INTERSECT_EPSILON*mag || t > (tmax + INTERSECT_EPSILON*mag))
		return false;

	TRI_FETCH(model, triID, 0, fetchedValue);
	// begin barycentric intersection algorithm
	unsigned int i1i2 = __float_as_int(fetchedValue.w) & 0xFFFF;
	hit.material = __float_as_int(fetchedValue.w) >> 16;

	p0 = __float_as_int(fetchedValue.x);
	p1 = __float_as_int(fetchedValue.y);
	p2 = __float_as_int(fetchedValue.z);

#	if 0
	VERT_FETCH(model, p0, 0, fetchedValue);
	//u0 = ((float*)&fetchedValue)[i1i2 & 0xFF];
	//v0 = ((float*)&fetchedValue)[i1i2 >> 8];;
	u0 = (i1i2 & 0xFF) == 0 ? fetchedValue.x : ((i1i2 & 0xFF) == 1 ? fetchedValue.y : fetchedValue.z);//
	v0 = (i1i2 >> 8) == 0 ? fetchedValue.x : ((i1i2 >> 8) == 1 ? fetchedValue.y : fetchedValue.z);//((float*)&fetchedValue)[i1i2 >> 8];
	VERT_FETCH(model, p1, 0, fetchedValue);
	//u1 = ((float*)&fetchedValue)[i1i2 & 0xFF];
	//v1 = ((float*)&fetchedValue)[i1i2 >> 8];
	u1 = (i1i2 & 0xFF) == 0 ? fetchedValue.x : ((i1i2 & 0xFF) == 1 ? fetchedValue.y : fetchedValue.z);//((float*)&fetchedValue)[i1i2 & 0xFF];
	v1 = (i1i2 >> 8) == 0 ? fetchedValue.x : ((i1i2 >> 8) == 1 ? fetchedValue.y : fetchedValue.z);//((float*)&fetchedValue)[i1i2 >> 8];
	VERT_FETCH(model, p2, 0, fetchedValue);
	//u2 = ((float*)&fetchedValue)[i1i2 & 0xFF];
	//v2 = ((float*)&fetchedValue)[i1i2 >> 8];
	u2 = (i1i2 & 0xFF) == 0 ? fetchedValue.x : ((i1i2 & 0xFF) == 1 ? fetchedValue.y : fetchedValue.z);//((float*)&fetchedValue)[i1i2 & 0xFF];
	v2 = (i1i2 >> 8) == 0 ? fetchedValue.x : ((i1i2 >> 8) == 1 ? fetchedValue.y : fetchedValue.z);//((float*)&fetchedValue)[i1i2 >> 8];

	u1 -= u0; 
	u2 -= u0; 
	u0 = ray.ori[i1i2 & 0xFF] + ray.dir[i1i2 & 0xFF] * t - u0; 

	v1 -= v0;
	v2 -= v0;
	v0 = ray.ori[i1i2 >> 8] + ray.dir[i1i2 >> 8] * t - v0;

	beta = (v0 * u1 - u0 * v1) / (v2 * u1 - u2 * v1);
	//if (beta < 0 || beta > 1)
	if (beta < -TRI_INTERSECT_EPSILON || beta > 1 + TRI_INTERSECT_EPSILON)
		return false;
	alpha = (u0 - beta * u2) / u1;	
	// not in triangle ?	
	if (alpha < -TRI_INTERSECT_EPSILON || (alpha + beta) > 1.0f + TRI_INTERSECT_EPSILON)
		return false;
#	else
	float4 _v0, _v1, _v2;
	VERT_FETCH(model, p0, 0, _v0);
	VERT_FETCH(model, p1, 0, _v1);
	VERT_FETCH(model, p2, 0, _v2);

	Vector3 v0 = Vector3(_v1 - _v0);
	Vector3 v1 = Vector3(_v2 - _v0);
	Vector3 v2 = Vector3(ray.ori + ray.dir*t - _v0);

	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float invDenom = 1.0f / (d00 * d11 - d01 * d01);
	alpha = (d11 * d20 - d01 * d21) * invDenom;
	beta = (d00 * d21 - d01 * d20) * invDenom;

	if (beta < -TRI_INTERSECT_EPSILON || beta > 1 + TRI_INTERSECT_EPSILON)
		return false;
	// not in triangle ?	
	if (alpha < -TRI_INTERSECT_EPSILON || (alpha + beta) > 1.0f + TRI_INTERSECT_EPSILON)
		return false;
#	endif

	VERT_FETCH(model, p0, 1, fetchedValue);
	float4 fetchedValue2;
	VERT_FETCH(model, p1, 1, fetchedValue2);
	hit.n.x = fetchedValue.x + alpha * (fetchedValue2.x - fetchedValue.x);
	hit.n.y = fetchedValue.y + alpha * (fetchedValue2.y - fetchedValue.y);
	hit.n.z = fetchedValue.z + alpha * (fetchedValue2.z - fetchedValue.z);

	VERT_FETCH(model, p2, 1, fetchedValue2);
	hit.n.x += beta * (fetchedValue2.x - fetchedValue.x);
	hit.n.y += beta * (fetchedValue2.y - fetchedValue.y);
	hit.n.z += beta * (fetchedValue2.z - fetchedValue.z);

	/*
	if(vdot > 0.0f)
		hit.n = -1.0f * hit.n;
	*/

	VERT_FETCH(model, p0, 3, fetchedValue);
	VERT_FETCH(model, p1, 3, fetchedValue2);
	hit.uv.x = fetchedValue.x + alpha * (fetchedValue2.x - fetchedValue.x);
	hit.uv.y = fetchedValue.y + alpha * (fetchedValue2.y - fetchedValue.y);

	VERT_FETCH(model, p2, 3, fetchedValue2);
	hit.uv.x += beta * (fetchedValue2.x - fetchedValue.x);
	hit.uv.y += beta * (fetchedValue2.y - fetchedValue.y);

	hit.model = model;
	/*
	hit.uv.x = VERT_FETCH_F(p0, 9) + 
		alpha * (VERT_FETCH_F(p1, 9)-VERT_FETCH_F(p0, 9)) + 
		beta * (VERT_FETCH_F(p2, 9)-VERT_FETCH_F(p0, 9));
	hit.uv.y = VERT_FETCH_F(p0, 10) + 
		alpha * (VERT_FETCH_F(p1, 10)-VERT_FETCH_F(p0, 10)) + 
		beta * (VERT_FETCH_F(p2, 10)-VERT_FETCH_F(p0, 10));
	*/
	/*
	hit.n.x = TRI_FETCH_F(triID, 3);
	hit.n.y = TRI_FETCH_F(triID, 4);
	hit.n.z = TRI_FETCH_F(triID, 5);
	*/

	//if((TRI_FETCH_F(triID, 3)*ray.dir.x + TRI_FETCH_F(triID, 4)*ray.dir.y + TRI_FETCH_F(triID, 5)*ray.dir.z) > 0.0f)
	//	hit.n = hit.n * -1.0f;
	//if(vdot > 0.0f) hit.n = hit.n * -1.0f;
	/*
	hit.n.x = vdot > 0.0f ? -hit.n.x : hit.n.x;
	hit.n.y = vdot > 0.0f ? -hit.n.y : hit.n.y;
	hit.n.z = vdot > 0.0f ? -hit.n.z : hit.n.z;
	*/
	getMaterial(hitMat, hit);

	// process transparent textures
	fetchedValue.w = 1.0f;
	if(hitMat.map_Ka.data)
	{
		TEX_FETCH(hit.model, hitMat.map_Ka, hit.uv.x, hit.uv.y, fetchedValue);
	}
	else if(hitMat.map_Kd.data)
	{
		TEX_FETCH(hit.model, hitMat.map_Kd, hit.uv.x, hit.uv.y, fetchedValue);
	}

	//float val = rnd(hit.seed);

	float val = rnd(seed);
	if(fetchedValue.w < val || hitMat.mat_Tf.x < val) return false;

	hit.t = t;
	tmax = t;
	hit.x = ray.ori + ray.dir * t;

	return true;
}

__device__ inline bool RayModelBVHIntersect(const Ray &oriRay, HitPoint &hit, Material &hitMat, int model, float tLimit, unsigned int seed)
{
	unsigned int stack[100];
	unsigned int stackPtr;
	//BSPArrayTreeNode currentNode;
	unsigned int left, right;
	float minx, miny, minz, maxx, maxy, maxz;
	float minT, maxT;
	unsigned int lChild;
	unsigned int axis;
	bool hasHit = false;

	Ray ray = oriRay;
	ray.transform(c_invTransfMatrix[model]);
	
	float mag = (fabsf(c_transfMatrix[model].x[0]) + fabsf(c_transfMatrix[model].x[5]) + fabsf(c_transfMatrix[model].x[10]))/3.0f;

	stack[0] = 0;
	stackPtr = 1;
	/*
	currentNode.low = tex1Dfetch(nodeTex, 0*2);
	currentNode.high = tex1Dfetch(nodeTex, 0*2+1);
	*/
	float4 fetchedData;
	NODE_FETCH(model, 0, 0, fetchedData);
	left = __float_as_int(fetchedData.x);
	right = __float_as_int(fetchedData.y);
	minx = fetchedData.z;
	miny = fetchedData.w;
	NODE_FETCH(model, 0, 1, fetchedData);
	minz = fetchedData.x;
	maxx = fetchedData.y;
	maxy = fetchedData.z;
	maxz = fetchedData.w;

	//if(model != 0)
	//{
	//	bool val = ray.BoxIntersect(minx, miny, minz, maxx, maxy, maxz, minT, maxT) && minT < hit.t && maxT > 0.000005f;
	//	printf("%d - %d : %f %f %f, %f %f %f, %f, %f\n",
	//		model, val, minx, miny, minz, maxx, maxy, maxz, minT, maxT);
	//}
	//	

	/*
	printf("currentNode : left = %d, right = %d, min = %f %f %f, max = %f %f %f\n", 
		currentNode.left, currentNode.right, currentNode.min[0], currentNode.min[1], currentNode.min[1], currentNode.max[0], currentNode.max[1], currentNode.max[1]);
	printf("ray : ori = %f %f %f, dir = %f %f %f\n", ray.ori[0], ray.ori[1], ray.ori[2], ray.dir[0], ray.dir[1], ray.dir[2]);
	*/

	//printf("%d %d\n%f %f %f\n%f %f %f\n", left, right, *(float*)&minx, *(float*)&miny, *(float*)&minz, *(float*)&maxx, *(float*)&maxy, *(float*)&maxz);
	//printf("%d %d %d\n%d %d %d\n", &minx, &miny, &minz, &maxx, &maxy, &maxz);
	for(;;)
	{
		if(ray.BoxIntersect(minx, miny, minz, maxx, maxy, maxz, minT, maxT) && minT < hit.t && maxT > 0.000005f)
		{
			if((left & 3) != 3)
			{
				lChild = (left >> 2);
				axis = (left & 3);
				stack[stackPtr] = (ray.posNeg[axis]) + lChild;
				/*
				currentNode.low = tex1Dfetch(nodeTex, ((ray.posNeg1[axis]^1) + lChild)*2);
				currentNode.high = tex1Dfetch(nodeTex, ((ray.posNeg1[axis]^1) + lChild)*2+1);
				*/
				NODE_FETCH(model, (ray.posNeg[axis]^1) + lChild, 0, fetchedData);
				left = __float_as_int(fetchedData.x);
				right = __float_as_int(fetchedData.y);
				minx = fetchedData.z;
				miny = fetchedData.w;
				NODE_FETCH(model, (ray.posNeg[axis]^1) + lChild, 1, fetchedData);
				minz = fetchedData.x;
				maxx = fetchedData.y;
				maxy = fetchedData.z;
				maxz = fetchedData.w;
				++stackPtr;
				continue;
			}
			else
			{
				hasHit = RayTriIntersect(ray, model, right, hit, hitMat, fminf(maxT, hit.t), mag, seed) || hasHit;

				if(tLimit > 0.0f && hasHit)
				{
					if(hit.t < tLimit) return true;
				}
			}
		}

		if (--stackPtr == 0) break;
		/*
		currentNode.low = tex1Dfetch(nodeTex, stack[stackPtr]*2);
		currentNode.high = tex1Dfetch(nodeTex, stack[stackPtr]*2+1);
		*/
		NODE_FETCH(model, stack[stackPtr], 0, fetchedData);
		left = __float_as_int(fetchedData.x);
		right = __float_as_int(fetchedData.y);
		minx = fetchedData.z;
		miny = fetchedData.w;
		NODE_FETCH(model, stack[stackPtr], 1, fetchedData);
		minz = fetchedData.x;
		maxx = fetchedData.y;
		maxy = fetchedData.z;
		maxz = fetchedData.w;
	}

	if(hasHit)
	{
		Vector3 hitX = ray.ori + ray.dir * hit.t;
		c_transfMatrix[model].transformPosition(hitX);
		int idx = oriRay.dir.indexOfMaxComponent();
		hit.t = (hitX.e[idx] - oriRay.ori.e[idx]) / oriRay.dir.e[idx];
		c_transfMatrix[model].transformVector(hit.n);
		hit.n = hit.n.normalize();
	}

	return hasHit;
}

__device__ inline bool RaySceneBVHIntersect(const Ray &ray, HitPoint &hit, Material &hitMat, float tLimit, unsigned int seed)
{
	unsigned int stack[100];
	unsigned int stackPtr;
	SceneNode *currentNode;
	float minT, maxT;
	bool hasHit = false;

	stack[0] = 0;
	stackPtr = 1;

	currentNode = &(c_scene.sceneGraph[0]);

	for(;;)
	{
		if(ray.BoxIntersect(currentNode->bbMin, currentNode->bbMax, minT, maxT) && minT < hit.t && maxT > 0.000005f)
		{
			if(currentNode->modelIdx >= 0)
			{
				hasHit = RayModelBVHIntersect(ray, hit, hitMat, currentNode->modelIdx, tLimit, seed) | hasHit;

				if(tLimit > 0.0f && hasHit)
				{
					if(hit.t < tLimit) return true;
				}
			}

			for(int i=0;i<currentNode->numChilds;i++)
			{
				stack[stackPtr++] = currentNode->childIdx[i];
			}
		}

		if (--stackPtr == 0) break;

		currentNode = &(c_scene.sceneGraph[stack[stackPtr]]);
	}
	return hasHit;
}

__host__ void unloadSceneCUDA()
{
	int device = 0;
	checkCudaErrors(cudaSetDevice(device));
	checkCudaErrors(cudaFree(d_cacheMem));
	checkCudaErrors(cudaFree(d_imageData));
	checkCudaErrors(cudaFree(d_summedImageData));
	checkCudaErrors(cudaFree(d_summedImageHitData));
	checkCudaErrors(cudaFree(d_summedImageDepthData));
	checkCudaErrors(cudaFree(d_summedImageNormalData));
	checkCudaErrors(cudaFree(d_envMapMem));
	h_image.width = h_image.height = 0;
}

__host__ void loadSceneCUDA(Scene *scene, bool reload = false)
{
	int device = 0;
	checkCudaErrors(cudaSetDevice(device));
	if(!reload) unloadSceneCUDA();

	h_scene = *scene;

	// geometry and materials
	checkCudaErrors(cudaMemcpyToSymbol(c_scene, scene, sizeof(Scene), 0, cudaMemcpyHostToDevice));

	size_t offsetModels[MAX_NUM_MODELS];
	size_t offsetVerts[MAX_NUM_MODELS];
	size_t offsetTris[MAX_NUM_MODELS];
	size_t offsetNodes[MAX_NUM_MODELS];
	size_t offsetMats[MAX_NUM_MODELS];
	size_t modelSize[MAX_NUM_MODELS];

	bool ignore[MAX_NUM_MODELS] = {0, };
	size_t modelOffset = 0, modelSizeTotal;
	for(int i=0;i<scene->numModels;i++)
	{
		modelSize[i] = 0;

		const Model &model = scene->models[i];
		modelSize[i] += model.numVerts*sizeof(Vertex);
		modelSize[i] += model.numTris*sizeof(Triangle);
		modelSize[i] += model.numNodes*sizeof(BVHNode);
		if(modelSize[i] > ((size_t)1)*1024*1024*1024)
		{
			//printf("Model[%d] is too big to load to GPU, ignore it\n", i);
			modelSize[i] = 0;
			ignore[i] = true;
		}
		if(!model.tris)
		{
			//printf("Model[%d] doesn't have data, ignore it\n", i);
			modelSize[i] = 0;
			ignore[i] = true;
		}
		modelSize[i] += model.numMats*sizeof(Material);
		for(int j=0;j<model.numMats;j++)
		{
			if(model.mats[j].map_Ka.data)
			{
				Texture &map = model.mats[j].map_Ka;
				modelSize[i] += map.width*map.height*4*sizeof(float);
			}
			if(model.mats[j].map_Kd.data)
			{
				Texture &map = model.mats[j].map_Kd;
				modelSize[i] += map.width*map.height*4*sizeof(float);
			}
			if(model.mats[j].map_bump.data)
			{
				Texture &map = model.mats[j].map_bump;
				modelSize[i] += map.width*map.height*4*sizeof(float);
			}
		}
		modelSize[i] += 16 - (modelSize[i] % 16);

		offsetModels[i] = modelOffset / 16;
		modelOffset += modelSize[i];
	}
	modelSizeTotal = modelOffset;

	if(!reload) checkCudaErrors(cudaMalloc((void **)&d_cacheMem, modelSizeTotal));

	for(int i=0;i<scene->numModels;i++)
	{
		size_t offset = 0;

		void *data = 0;
		size_t size = 0;

		const Model &model = scene->models[i];

		if(!ignore[i])
		{
			for(int j=0;j<3;j++)
			{
				switch(j)
				{
				case 0: data = model.verts;	size = model.numVerts*sizeof(Vertex);	offsetVerts[i] = offset/16;	break;
				case 1: data = model.tris;	size = model.numTris*sizeof(Triangle);	offsetTris[i] = offset/16;	break;
				case 2: data = model.nodes;	size = model.numNodes*sizeof(BVHNode);	offsetNodes[i] = offset/16;	break;
				}
				checkCudaErrors(cudaMemcpy((void*)(d_cacheMem+offsetModels[i]*16+offset), data, size, cudaMemcpyHostToDevice));
				offset += size;
			}
		}

		// texture mapping
		for(int j=0;j<model.numMats;j++)
		{
			Texture &map_Ka = model.mats[j].map_Ka;
			if(map_Ka.data)
			{
				data = map_Ka.data;
				size = map_Ka.width*map_Ka.height*4*sizeof(float);
				checkCudaErrors(cudaMemcpy((void*)(d_cacheMem+offsetModels[i]*16+offset), data, size, cudaMemcpyHostToDevice));
				map_Ka.offset = (int)(offset/16);
				offset += size;
			}

			Texture &map_Kd = model.mats[j].map_Kd;
			if(map_Kd.data)
			{
				data = map_Kd.data;
				size = map_Kd.width*map_Kd.height*4*sizeof(float);
				checkCudaErrors(cudaMemcpy((void*)(d_cacheMem+offsetModels[i]*16+offset), data, size, cudaMemcpyHostToDevice));
				map_Kd.offset = (int)(offset/16);
				offset += size;
			}

			Texture &map_bump = model.mats[j].map_bump;
			if(map_bump.data)
			{
				data = map_bump.data;
				size = map_bump.width*map_bump.height*4*sizeof(float);
				checkCudaErrors(cudaMemcpy((void*)(d_cacheMem+offsetModels[i]*16+offset), data, size, cudaMemcpyHostToDevice));
				map_bump.offset = (int)(offset/16);
				offset += size;
			}
		}

		// general materials
		data = model.mats;	size = model.numMats*sizeof(Material);	offsetMats[i] = offset/16;
		checkCudaErrors(cudaMemcpy((void*)(d_cacheMem+offsetModels[i]*16+offset), data, size, cudaMemcpyHostToDevice));
		offset += size;

		for(int j=0;j<16;j++)
		{
			h_transfMatrix[i][j] = model.transfMatrix[j];
			h_invTransfMatrix[i][j] = model.invTransfMatrix[j];
		}
	}

	//unsigned char *temp = new unsigned char[offsetModels[2]*16];
	//cudaMemcpy((void*)temp, d_cacheMem, offsetModels[2]*16, cudaMemcpyDeviceToHost);

	for(int i=0;i<scene->numModels;i++)
	{
		h_offsetModels[i] = offsetModels[i];
		h_offsetVerts[i] = offsetVerts[i];
		h_offsetTris[i] = offsetTris[i];
		h_offsetNodes[i] = offsetNodes[i];
		h_offsetMats[i] = offsetMats[i];
	}

	checkCudaErrors(cudaBindTexture((size_t *)0, t_model, d_cacheMem, modelSizeTotal));

	checkCudaErrors(cudaMemcpyToSymbol(c_offsetModels, offsetModels, sizeof(size_t)*MAX_NUM_MODELS, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_offsetVerts, offsetVerts, sizeof(size_t)*MAX_NUM_MODELS, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_offsetTris, offsetTris, sizeof(size_t)*MAX_NUM_MODELS, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_offsetNodes, offsetNodes, sizeof(size_t)*MAX_NUM_MODELS, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_offsetMats, offsetMats, sizeof(size_t)*MAX_NUM_MODELS, 0, cudaMemcpyHostToDevice));

	// emitters
	checkCudaErrors(cudaMemcpyToSymbol(c_emitterList, scene->emitters, sizeof(Emitter)*min(scene->numEmitters, MAX_NUM_EMITTERS), 0, cudaMemcpyHostToDevice));

	// transform matrix
	checkCudaErrors(cudaMemcpyToSymbol(c_transfMatrix, h_transfMatrix, sizeof(Matrix)*min(scene->numModels, MAX_NUM_MODELS), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_invTransfMatrix, h_invTransfMatrix, sizeof(Matrix)*min(scene->numModels, MAX_NUM_MODELS), 0, cudaMemcpyHostToDevice));

	bool hasEvnMap = false;

	if(scene->envMap[0].data)
	{
		// environment map
		int sizeTex = scene->envMap[0].width*scene->envMap[0].height*scene->envMap[0].bpp*sizeof(float);
		if(!reload) checkCudaErrors(cudaMalloc((void **)&d_envMapMem, sizeTex*6));
		size_t offset = 0;
		for(int i=0;i<6;i++)
		{
			Image &map = scene->envMap[i];
			checkCudaErrors(cudaMemcpy((void*)((unsigned char*)d_envMapMem+offset), map.data, sizeTex, cudaMemcpyHostToDevice));
			offset += sizeTex;
		}
		checkCudaErrors(cudaBindTexture((size_t *)0, t_envMap, d_envMapMem, sizeTex*6));
		
		hasEvnMap = true;
	}
	checkCudaErrors(cudaMemcpyToSymbol(c_envCol, &(scene->envColor), sizeof(Vector3), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_hasEnvMap, &hasEvnMap, sizeof(bool), 0, cudaMemcpyHostToDevice));
}

__host__ void materialChangedCUDA(Scene *scene)
{
	for(int i=0;i<scene->numModels;i++)
	{
		const Model &model = scene->models[i];

		void *data = 0;
		size_t size = 0;

		// general materials
		data = model.mats;	size = model.numMats*sizeof(Material);
		checkCudaErrors(cudaMemcpy((void*)(d_cacheMem+h_offsetModels[i]*16+h_offsetMats[i]*16), data, size, cudaMemcpyHostToDevice));
	}
}

__host__ void lightChangedCUDA(Scene *scene)
{
	h_scene = *scene;
	checkCudaErrors(cudaMemcpyToSymbol(c_scene, scene, sizeof(Scene), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_emitterList, scene->emitters, sizeof(Emitter)*min(scene->numEmitters, MAX_NUM_EMITTERS), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_envCol, &(scene->envColor), sizeof(Vector3), 0, cudaMemcpyHostToDevice));
}

__host__ void updateController(Controller *controller)
{
	if(controller)
	{
		h_controller = *controller;
		checkCudaErrors(cudaMemcpyToSymbol(c_controller, controller, sizeof(Controller), 0, cudaMemcpyHostToDevice));
	}
}

__host__ void clearResult(int &frame)
{
	checkCudaErrors(cudaMemset(d_summedImageData, 0, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
	checkCudaErrors(cudaMemset(d_summedImageHitData, 0, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
	checkCudaErrors(cudaMemset(d_summedImageDepthData, 0, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
	checkCudaErrors(cudaMemset(d_summedImageNormalData, 0, h_image.width*h_image.height*h_image.bpp*sizeof(float)));

	frame = 1;
	h_frame = 1;
}


__host__ void renderBeginCUDA(Camera *camera, Image *image, Controller *controller, int &frame)
{
	int device = 0;
	int oldDevice;
	cudaGetDevice(&oldDevice);
	if(device != oldDevice)
		cudaSetDevice(device);

	if(*camera != h_camera)
	{
		frame = 1;
	}

	if(camera)
	{
		h_camera = *camera;
		checkCudaErrors(cudaMemcpyToSymbol(c_camera, camera, sizeof(Camera), 0, cudaMemcpyHostToDevice));
	}

	updateController(controller);

	if(image->width != h_image.width || image->height != h_image.height || image->bpp != h_image.bpp)
	{
		checkCudaErrors(cudaFree(d_imageData));
		checkCudaErrors(cudaFree(d_summedImageData));
		checkCudaErrors(cudaFree(d_summedImageHitData));
		checkCudaErrors(cudaFree(d_summedImageDepthData));
		checkCudaErrors(cudaFree(d_summedImageNormalData));
		h_image = *image;

		checkCudaErrors(cudaMalloc((void**)&d_imageData, h_image.width*h_image.height*h_image.bpp));
		checkCudaErrors(cudaMalloc((void**)&d_summedImageData, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_summedImageHitData, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_summedImageDepthData, h_image.width*h_image.height*h_image.bpp*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_summedImageNormalData, h_image.width*h_image.height*h_image.bpp*sizeof(float)));

		checkCudaErrors(cudaMemcpyToSymbol(c_imgWidth, &h_image.width, sizeof(int), 0, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyToSymbol(c_imgHeight, &h_image.height, sizeof(int), 0, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyToSymbol(c_imgBpp, &h_image.bpp, sizeof(int), 0, cudaMemcpyHostToDevice));
		
	}

	h_frame = frame;
}

__host__ void renderEndCUDA(Image *image)
{
	checkCudaErrors(cudaMemcpy(image->data, d_imageData, h_image.width*h_image.height*h_image.bpp, cudaMemcpyDeviceToHost));
	//cudaMemcpy(image->data, d_imageData, h_image.width*h_image.height*h_image.bpp, cudaMemcpyDeviceToHost);
	//cudaMemcpy(image->data, d_imageData, 100, cudaMemcpyDeviceToHost);
}