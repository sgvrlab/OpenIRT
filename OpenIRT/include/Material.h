#pragma once

#include "RGB.h"
#include "BitmapTexture.h"
#include "random.h"

namespace irt
{

class Material
{
public:
	Material()
	{ 
		setDefault();
	};

	~Material() {clear();};

	void clear()
	{
		mat_Ka = RGBf(0.0f, 0.0f, 0.0f);
		mat_Kd = RGBf(0.0f, 0.0f, 0.0f);
		mat_Ks = RGBf(0.0f, 0.0f, 0.0f);
		mat_Tf = RGBf(0.0f, 0.0f, 0.0f);
		mat_d = 0.f;
		mat_Ns = 0.f;
		mat_illum = 2; //default

		//if(map_Ka) delete map_Ka;
		//if(map_Kd) 
		//	delete map_Kd;
		//if(map_bump) delete map_bump;

		name[0] = 0;

		rangeKd = 0.f;
		rangeKs = 0.f;

		map_Ka = 0;
		map_Kd = 0;
		map_bump = 0;
	}

	void setDefault()
	{
		map_Ka = 0;
		map_Kd = 0;
		map_bump = 0;

		clear();
		mat_Kd = RGBf(1.0f, 1.0f, 1.0f);	// default material
		mat_Tf = RGBf(1.0f, 1.0f, 1.0f);
		mat_d = 1.0f;

		rangeKd = rangeKs = 1.0f;
	}

	bool hasDiffuse() {return mat_Kd > RGBf(0,0,0);}
	bool hasSpecular() {return mat_Ks > RGBf(0,0,0);}
	
	void setMatKa(const RGBf &newKa) {mat_Ka = newKa;}
	void setMatKd(const RGBf &newKd) {mat_Kd = newKd;recalculateRanges();}
	void setMatKs(const RGBf &newKs) {mat_Ks = newKs;recalculateRanges();}
	void setMatTf(const RGBf &newTf) {mat_Tf = newTf;}
	void setMat_d(const float newMat_d) {mat_d = newMat_d;}
	void setMat_Ns(const float newMat_Ns) {mat_Ns = newMat_Ns;}
	void setMat_illum(const int illum) {mat_illum = illum;}
	void setName( const char *matName ) {strcpy_s(name, 59, matName);}
	void setMapKa(const char *mapFileName);
	void setMapKd(const char *mapFileName);
	void setMapBump( const char *mapFileName );

	const RGBf &getMatKa() const {return mat_Ka;}
	const RGBf &getMatKd() const {return mat_Kd;}
	const RGBf &getMatKs() const {return mat_Ks;}
	const RGBf &getMatTf() const {return mat_Tf;}
	float getMat_d() const {return mat_d;}
	float getMat_Ns() const {return mat_Ns;}
	int getMat_illum() const {return mat_illum;}
	const char *getName() const {return name;}
	BitmapTexture *getMapKa() const {return map_Ka;}
	BitmapTexture *getMapKd() const {return map_Kd;}
	BitmapTexture *getMapBump() const {return map_bump;}

	bool operator == (const Material &mat)
	{
		if(mat_Ka != mat.mat_Ka) return false;
		if(mat_Kd != mat.mat_Kd) return false;
		if(mat_Ks != mat.mat_Ks) return false;
		if(mat_Tf != mat.mat_Tf) return false;
		if(mat_d != mat.mat_d) return false;
		if(mat_Ns != mat.mat_Ns) return false;
		if(mat_illum != mat.mat_illum) return false;
		if(map_Ka != mat.map_Ka) return false;
		if(map_Kd != mat.map_Kd) return false;
		if(map_bump != mat.map_bump) return false;
		return true;
	}

	bool operator != (const Material &mat)
	{
		return !(*this == mat);
	}

	bool isPerfectSpecular(float mat_Ns) const {return rangeKs != rangeKd && mat_Ns > 2047.0f;}
	bool isPerfectSpecular() const {return rangeKs != rangeKd && mat_Ns > 2047.0f;}

	void recalculateRanges()
	{
		rangeKd = mat_Kd.r() + mat_Kd.g() + mat_Kd.b();
		rangeKs = mat_Ks.r() + mat_Ks.g() + mat_Ks.b();
		rangeKs += rangeKd;
	}

	static inline bool isDiffuse(float rangeKd, float rangeKs, unsigned int seed) {return rnd(seed)*rangeKs <= rangeKd;}
	static inline bool isRefraction(float mat_d, unsigned int seed) {return rnd(seed) >= mat_d;}

	inline bool isDiffuse(unsigned int seed) {return isDiffuse(rangeKd, rangeKs, seed);}
	inline bool isRefraction(unsigned int seed) {return isRefraction(mat_d, seed);}

	static Vector3 sampleDiffuseDirection(const Vector3 &normal, unsigned int &prevRnd)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * 3.141592f * rnd(prevRnd);
		float r = sqrtf(rnd(prevRnd));
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = cross(normal, m1);
		if (U.length() < 0.01f)
			U = cross(normal, m2); 
		Vector3 V = cross(normal, U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
	}

	static Vector3 sampleAmbientOcclusionDirection(const Vector3 &normal, unsigned int &prevRnd)
	{
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * 3.141592f * rnd(prevRnd);
		float r = sqrtf(rnd(prevRnd)*0.95f);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = cross(normal, m1);
		if (U.length() < 0.01f)
			U = cross(normal, m2); 
		Vector3 V = cross(normal, U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
	}

	Vector3 sampleDirection(const Vector3 &normal, const Vector3 &inDirection, unsigned int seed)
	{
		/*
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * 3.141592f * rnd(prevRnd);
		float r = sqrtf(rnd(prevRnd));
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = cross(normal, m1);
		if (U.length() < 0.01f)
			U = cross(normal, m2); 
		Vector3 V = cross(normal, U);

		// use coordinates in basis:
		return x * U + y * V + z * normal;
		*/
		Vector3 m1(1.0f, 0.0f, 0.0f);
		Vector3 m2(0.0f, 1.0f, 0.0f);
		Vector3 W(0.0f, 0.0f, 1.0f);

		float x = 0.0f, y = 0.0f, z = 1.0f;

		bool isDiff = isDiffuse(rangeKd, rangeKs, seed);
		bool isRefrac = isRefraction(mat_d, seed);

		if(isDiff)
		{
			// diffuse reflection

			W = normal;

			float phi = 2.0f * PI * rnd(seed);
			float r = sqrtf(rnd(seed));
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
				if(!isPerfectSpecular())
				{
					float phi = 2.0f * PI * rnd(seed);

					float cosTheta = powf((1.0f - rnd(seed)), 1.0f / mat_Ns);
					float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

					x = sinTheta * cosf(phi);
					y = sinTheta * sinf(phi);
					z = cosTheta;
				}

				W = inDirection - 2.0f*dot(inDirection, normal)*normal;
			}
			else
			{
				/*
				// refraction

				float dn = dot(inDirection, normal);
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
		Vector3 U = cross(W, m1);
		if (U.length() < 0.01f)
			U = cross(W, m2); 
		Vector3 V = cross(W, U);

		// use coordinates in basis:
		return x * U + y * V + z * W;
	}

	inline RGBf brdf(unsigned int seed, bool isDiffuse)
	{
		// assumed that outgoing direction is a result of sampleDirection
		return isDiffuse ? mat_Kd : mat_Ks;
	}

	inline RGBf brdf(unsigned int seed)
	{
		return brdf(seed, isDiffuse(seed));
	}
//protected:
public:
	
	RGBf mat_Ka;			// ambient reflectance
	RGBf mat_Kd;			// diffuse	reflectance
	RGBf mat_Ks;			// specular reflectance
	RGBf mat_Tf;			// transmission filter
	float mat_d;			// dissolve, (1(default): opaque, 0: transparent)
	float mat_Ns;			// specular exponent
	int mat_illum;			// illumination model

	char name[60];

	float rangeKd;
	float rangeKs;

	BitmapTexture *map_Ka;		// bitmap ambient
	BitmapTexture *map_Kd;		// bitmap diffuse
	BitmapTexture *map_bump;	// bitmap bump
};

typedef std::vector<Material> MaterialList;
bool loadMaterialFromMTL(const char *fileName, MaterialList &matList);
bool saveMaterialToMTL(const char *fileName, MaterialList &matList);
bool generateMTLFromOOCMaterial(const char *oocFileName, const char *mtlFileName);
void modifyMaterial(MaterialList &matList, Material which, Material to);

};