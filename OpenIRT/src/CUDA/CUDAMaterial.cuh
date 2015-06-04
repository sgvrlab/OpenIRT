#ifndef CUDA_MATERIAL_H
#define CUDA_MATERIAL_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "CUDAVector.cuh"
#include "CUDARandom.cuh"

namespace CUDA
{

typedef cuVec3 Vector3;
typedef cuVec2 Vector2;

typedef union Texture_t {
	struct {
		unsigned short width;
		unsigned short height;
		int offset;
		float *data;
	};

	struct {
		float4 quad;
	};

	__device__ __host__ Texture_t()
	{
		width = height = 0;
		offset = 0;
		data = 0;
	}

	__device__ inline int getTexOffset(const float &u, const float &v)
	{
		int x = (int)(u*(float)width) % width;
		int y = (int)(v*(float)height) % height;

		x = x < 0 ? width + x : x;
		y = y < 0 ? height + y : y;

		y = height - y - 1;

		return offset+y*width+x;
	}
} Texture;


typedef union Material_t {
	struct {
	Vector3 mat_Ka;			// ambient reflectance
	Vector3 mat_Kd;			// diffuse	reflectance
	Vector3 mat_Ks;			// specular reflectance
	Vector3 mat_Tf;			// transmission filter
	float mat_d;			// dissolve, (1(default): opaque, 0: transparent)
	float mat_Ns;			// specular exponent
	int mat_illum;			// illumination model

	char name[60];

	float rangeKd;
	float rangeKs;

	Texture map_Ka;
	Texture map_Kd;
	Texture map_bump;	// bitmap bump
	};

	struct {
		float4 _0;
		float4 _1;
		float4 _2;
		float4 _3;
		float4 _4;
		float4 _5;
		float4 _6;
		float4 _7;
		float4 _8;
		float4 _9;
		float4 _10;
	};

	__device__ static Vector3 sampleDeterminedDiffuseDirection(const Vector3 &normal, unsigned int iter)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * 3.141592f * ((iter%8)+0.5f)/8.0f;
		float r = sqrtf(((iter/8)+0.5f)/2.0f);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = normal.cross(m1);
		if (U.length() < 0.01f)
			U = normal.cross(m2); 
		Vector3 V = normal.cross(U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
	}

	__device__ static Vector3 sampleDiffuseDirection(const Vector3 &normal, unsigned int prevRnd)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * 3.141592f * rnd(prevRnd);
		float r = sqrtf(rnd(prevRnd)*0.9f);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = normal.cross(m1);
		if (U.length() < 0.01f)
			U = normal.cross(m2); 
		Vector3 V = normal.cross(U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
	}

	__device__ static Vector3 sampleDiffuseDirection2(const Vector3 &normal, unsigned int &prevRnd, float maxR)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * 3.141592f * rnd(prevRnd);
		float r = sqrtf(rnd(prevRnd)*maxR);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = normal.cross(m1);
		if (U.length() < 0.01f)
			U = normal.cross(m2); 
		Vector3 V = normal.cross(U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
	}

	__device__ static Vector3 sampleDiffuseDirectionWithJitter(const Vector3 &normal, unsigned int prevRnd, unsigned int jitterRnd)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float val1 = rnd(prevRnd) + (rnd(jitterRnd) - 0.5f)*1.0f;
		float val2 = rnd(prevRnd) + (rnd(jitterRnd) - 0.5f)*1.0f;

		if(val1 < 0.0f) val1 = 0.0f;
		if(val2 < 0.0f) val2 = 0.0f;
		if(val1 >= 1.0f) val1 = 0.999999f;
		if(val2 >= 1.0f) val2 = 0.999999f;

		float phi = 2.0f * 3.141592f * val1;
		float r = sqrtf(val2*0.9f);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = normal.cross(m1);
		if (U.length() < 0.01f)
			U = normal.cross(m2); 
		Vector3 V = normal.cross(U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
	}

	__device__ static Vector3 sampleAmbientOcclusionDirection(const Vector3 &normal, unsigned int prevRnd)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * 3.141592f * rnd(prevRnd);
		float r = sqrtf(rnd(prevRnd)*0.95f);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = normal.cross(m1);
		if (U.length() < 0.01f)
			U = normal.cross(m2); 
		Vector3 V = normal.cross(U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
	}

	__device__ __host__ inline bool hasDiffuse() {return rangeKd > 0.0f;}

	__device__ __host__ static inline bool isDiffuse(float rangeKd, float rangeKs, unsigned int seed) {return rnd(seed)*rangeKs <= rangeKd;}
	__device__ __host__ static inline bool isRefraction(float mat_d, unsigned int seed) {return rnd(seed) >= mat_d;}

	__device__ __host__ inline bool isDiffuse(unsigned int seed) {return isDiffuse(rangeKd, rangeKs, seed);}
	__device__ __host__ inline bool isRefraction(unsigned int seed) {return isRefraction(mat_d, seed);}

	__device__ __host__ inline void recalculateRanges()
	{
		rangeKd = mat_Kd.x + mat_Kd.y + mat_Kd.z;
		rangeKs = mat_Ks.x + mat_Ks.y + mat_Ks.z;
		rangeKs += rangeKd;
	}

	__device__ __host__ static Vector3 sampleDirection(float rangeKd, float rangeKs, float mat_d, float mat_Ns, const Vector3 &normal, const Vector3 &inDirection, unsigned int seed)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);
		Vector3 W(0.0f, 0.0f, 1.0f);

		float x = 0.0f, y = 0.0f, z = 1.0f;

		bool isDiff = isDiffuse(rangeKd, rangeKs, seed);
		bool isRefrac = isRefraction(mat_d, seed);

		float val1 = rnd(seed);
		float val2 = rnd(seed);

		if(isDiff)
		{
			// diffuse reflection
			W = normal;

			float phi = 2.0f * PI * val1;
			float r = sqrtf(val2);
			x = r * cosf(phi);
			y = r * sinf(phi);
			z = sqrtf(1.0f - x*x - y*y);
		}
		else
		{
			// specular reflection or refraction

			if(!isRefrac)
			{
				// reflection
				if(mat_Ns < 2048)
				{
					float phi = 2.0f * PI * val1;

					float cosTheta = powf((1.0f - val2), 1.0f / mat_Ns);
					float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

					x = sinTheta * cosf(phi);
					y = sinTheta * sinf(phi);
					z = cosTheta;
				}

				W = inDirection - 2.0f*inDirection.dot(normal)*normal;
			}
			else
			{
				/*
				// refraction

				float dn = inDirection.dot(normal);
				float indexOfRefraction = 1.46f;	// assuming glass, use 1.3 for water

				if(dn < 0.0f)
				{
					// incoming ray
					float temp = 1.0f / indexOfRefraction;
					dn = -dn;
					float root = 1.0f - (temp*temp) * (1.0f - dn*dn);
					W = inDirection*temp + normal*(temp*dn - sqrt(root));
				}
				else
				{
					// outgoing ray
					float root = 1.0f - (indexOfRefraction*indexOfRefraction) * (1.0f - dn*dn);
					if(root >= 0.0f)
						W = inDirection*indexOfRefraction - normal*(indexOfRefraction*dn - sqrt(root));
				}
				*/
				W = inDirection;
			}
		}

		// build orthonormal basis from W
		Vector3 U = W.cross(m1);
		if (U.length() < 0.01f)
			U = W.cross(m2); 
		Vector3 V = W.cross(U);

		// use coordinates in basis:
		return x * U + y * V + z * W;
	}

	__device__ __host__ static Vector3 sampleDirectionWithJitter(float rangeKd, float rangeKs, float mat_d, float mat_Ns, const Vector3 &normal, const Vector3 &inDirection, unsigned int seed, unsigned int jitterRnd)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);
		Vector3 W(0.0f, 0.0f, 1.0f);

		float x = 0.0f, y = 0.0f, z = 1.0f;

		bool isDiff = isDiffuse(rangeKd, rangeKs, seed);
		bool isRefrac = isRefraction(mat_d, seed);

		float val1 = rnd(jitterRnd);
		float val2 = rnd(jitterRnd);

		if(isDiff)
		{
			// diffuse reflection
			W = normal;

			float phi = 2.0f * PI * val1;
			float r = sqrtf(val2);
			x = r * cosf(phi);
			y = r * sinf(phi);
			z = sqrtf(1.0f - x*x - y*y);
		}
		else
		{
			// specular reflection or refraction

			if(!isRefrac)
			{
				// reflection
				if(mat_Ns < 2048)
				{
					float phi = 2.0f * PI * val1;

					float cosTheta = powf((1.0f - val2), 1.0f / mat_Ns);
					float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

					x = sinTheta * cosf(phi);
					y = sinTheta * sinf(phi);
					z = cosTheta;
				}

				W = inDirection - 2.0f*inDirection.dot(normal)*normal;
			}
			else
			{
				/*
				// refraction

				float dn = inDirection.dot(normal);
				float indexOfRefraction = 1.46f;	// assuming glass, use 1.3 for water

				if(dn < 0.0f)
				{
					// incoming ray
					float temp = 1.0f / indexOfRefraction;
					dn = -dn;
					float root = 1.0f - (temp*temp) * (1.0f - dn*dn);
					W = inDirection*temp + normal*(temp*dn - sqrt(root));
				}
				else
				{
					// outgoing ray
					float root = 1.0f - (indexOfRefraction*indexOfRefraction) * (1.0f - dn*dn);
					if(root >= 0.0f)
						W = inDirection*indexOfRefraction - normal*(indexOfRefraction*dn - sqrt(root));
				}
				*/
				W = inDirection;
			}
		}

		// build orthonormal basis from W
		Vector3 U = W.cross(m1);
		if (U.length() < 0.01f)
			U = W.cross(m2); 
		Vector3 V = W.cross(U);

		// use coordinates in basis:
		return x * U + y * V + z * W;
	}

	//__device__ Vector3 sampleDirection(const Vector3 &normal, const Vector3 &inDirection, unsigned int seed, unsigned int seed2)
	__device__ inline Vector3 sampleDirection(const Vector3 &normal, const Vector3 &inDirection, unsigned int seed)
	{
		return sampleDirection(rangeKd, rangeKs, mat_d, mat_Ns, normal, inDirection, seed);
	}

	//__device__ Vector3 brdf(const Vector3 &normal, const Vector3 &inDirection, const Vector3 &outDirection, unsigned int seed)
	__device__ inline Vector3 brdf(unsigned int seed, bool isDiffuse)
	{
		// assumed that outgoing direction is a result of sampleDirection
		return isDiffuse ? mat_Kd : mat_Ks;
		/*
		if(rnd(seed)*rangeKs <= rangeKd)
		{
			// diffuse reflection
			return mat_Kd;// / PI;
		}
		else
		{
			// specular reflection or refraction
			return mat_Ks;
		}
		*/
	}

	__device__ inline Vector3 brdf(unsigned int seed)
	{
		return brdf(seed, isDiffuse(seed));
	}
} Material;

typedef struct MaterialSampler_t {
	struct
	{
		float rangeKd;
		float rangeKs;
		float mat_d;
		float mat_Ns;
	};

	__device__ inline void getSampler(const Material &mat)
	{
		rangeKd = mat.rangeKd;
		rangeKs = mat.rangeKs;
		mat_d = mat.mat_d;
		mat_Ns = mat.mat_Ns;
	}

	__host__ __device__ void reset()
	{
		rangeKd = rangeKs = mat_d = mat_Ns = 0.0f;
	}
} MaterialSampler;

typedef struct Parallelogram_t {
	struct {
	Vector3 corner;			
	Vector3 v1;
	Vector3 v2;
	Vector3 normal;
	};

	__device__ inline Vector3 sample(unsigned int prevRnd) const
	{
		return corner + v1 * rnd(prevRnd) + v2 * rnd(prevRnd);
	}

	__device__ inline Vector3 sample(unsigned int prevRnd, unsigned int prevRnd2) const
	{
		return corner + v1 * rnd2(prevRnd, prevRnd2) + v2 * rnd2(prevRnd, prevRnd2);
	}
} Parallelogram;

typedef struct Emitter_t {
	enum Type
	{
		POINT_LIGHT,
		TRIANGULAR_LIGHT,
		PARALLELOGRAM_LIGHT,
		CAMERA_LIGHT,
		ENVIRONMENT_LIGHT
	};

	enum TargetType
	{
		LIGHT_TARGET_NONE,
		LIGHT_TARGET_SPHERE,
		LIGHT_TARGET_HALF_SPHERE,
		LIGHT_TARGET_PARALLELOGRAM
	};

	struct {
	Vector3 color_Ka;		// ambient light
	Vector3 color_Kd;		// diffuse light
	Vector3 color_Ks;		// specular light
	Type type;				// light type
	TargetType targetType;	// light target type

	char name[60];

	Vector3 pos;
	Parallelogram planar;
	Parallelogram spotTarget;
	int numScatteringPhotons;
	float intensity;
	bool isCosLight;

	char environmentTexName[256];

	};

	__device__ inline Vector3 sample(unsigned int prevRnd) const
	{
		if(type == PARALLELOGRAM_LIGHT)
			return planar.sample(prevRnd);
		return pos;
	}

	__device__ inline Vector3 sample(unsigned int prevRnd, unsigned int prevRnd2) const
	{
		if(type == PARALLELOGRAM_LIGHT)
			return planar.sample(prevRnd, prevRnd2);
		return pos;
	}

	__device__ inline Vector3 sampleEmitDirection(const Vector3 &ori, unsigned int prevRnd) const
	{
		/*
		if(type == PARALLELOGRAM_LIGHT)
		{
			if(spotTarget.corner.x == 3.402823466e+38F)
				return Material::sampleDiffuseDirection(planar.normal, prevRnd);
			else
				return spotTarget.sample(prevRnd) - ori;
		}
		return (Vector3(rnd(prevRnd), rnd(prevRnd), rnd(prevRnd))).normalize();
		*/
		switch(targetType)
		{
		case LIGHT_TARGET_SPHERE : return Vector3(rnd(prevRnd) - 0.5f, rnd(prevRnd) - 0.5f, rnd(prevRnd) - 0.5f).normalize();
		case LIGHT_TARGET_HALF_SPHERE : return Material::sampleDiffuseDirection(planar.normal, prevRnd);
		case LIGHT_TARGET_PARALLELOGRAM : return (spotTarget.sample(prevRnd) - ori).normalize();
		}
		return Vector3(0.0f);
	}
} Emitter;

}

#endif // #ifndef CUDA_MATERIAL_H
