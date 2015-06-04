#ifndef COMMON_MATERIALBITMAPTEXTURE_H
#define COMMON_MATERIALBITMAPTEXTURE_H

#include "Materials.h"
#include "Noise.h"
#include "TextureManager.h"

/**
* Material with bitmapped texture
*/
class MaterialBitmapTexture : public Material
{
public:	
	MaterialBitmapTexture (const char *textureFileName) { 
		TextureManager *texMan = TextureManager::getSingletonPtr();

		texPtr = texMan->loadTexture(textureFileName);

		if (!texPtr) {
			LogManager *log = LogManager::getSingletonPtr();						
			log->logMessage(LOG_ERROR, texMan->getLastErrorString());
		}
	}

	~MaterialBitmapTexture () {
		TextureManager *texMan = TextureManager::getSingletonPtr();

		if (texPtr) {
			texMan->unloadTexture(texPtr);
		}
	} 

	/*
	virtual void shade(Vector3 &viewer, Hitpoint &p, rgb &result);		

	virtual void shade(Vector3 &viewer, SIMDHitpoint &p, int idx, rgb& result);
	*/
	void shade(Vector3 &viewer, Hitpoint &p, rgb &result) {}

	void shade(Vector3 &viewer, SIMDHitpoint &p, int idx, rgb& result) {}
	
	// For explicitly sampling the BRDF of the material. vIn and vOut
	// must be normalized.
	virtual void brdf(const Hitpoint &p, const Vector3 &vIn, const Vector3 &vOut, rgb &brdfResult) {
		if (texPtr != 0) {
			// BRDF for perfect diffuse is 1/PI:
			texPtr->getTexValue(brdfResult, p.uv[0], p.uv[1]);
			brdfResult /= PI;

		}
	}

	/** 
	* Get a direction to sample further. Direction vIn mus be normalized. The function
	* returns the reflectance for the direction that was sampled. The sampled direction is
	* written into newDirection.
	*
	* This samples a direction for a perfect diffuse material.
	*/
	virtual void sampleDirection(const Hitpoint &p, const Vector3 &vIn, Vector3 &newDirection, rgb &reflectance) {
		static Vector3 m1(1.0f, 0.0f, 0.0f);
		static Vector3 m2(0.0f, 1.0f, 0.0f);

		float phi = 2.0f * PI * materialSampler.sample();
		float r = sqrtf(materialSampler.sample());
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from n
		Vector3 U = cross(p.n, m1);
		if (U.length() < 0.01f)
			U = cross(p.n, m2); 
		Vector3 V = cross(p.n, U);

		// use coordinates in basis:
		newDirection = x * U + y * V + z * p.n;
		
		// return probability of reflection (rgb color times constant factor currently)
		texPtr->getTexValue(reflectance, p.uv[0], p.uv[1]);
		reflectance *= MATERIAL_DIFFUSE_MAXREFLECTANCE;
	}

protected:

	// bitmap we're using for shading
	BitmapTexture *texPtr;

private:
};

#endif