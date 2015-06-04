#ifndef COMMON_MATERIALSOLIDNOISE_H
#define COMMON_MATERIALSOLIDNOISE_H

#include "Materials.h"
#include "Noise.h"
#include "ONB.h"

/**
* Material with bitmapped texture
*/
class MaterialSolidNoise : public Material
{
public:	
	MaterialSolidNoise () { 
		c0 = rgb(0.8f,0,0); 
		c1 = rgb(0.0f,0,0.8f); 
		scale = 0.5f;
	}

	~MaterialSolidNoise () {} ;

	void shade(Vector3 &viewer, Hitpoint &p, rgb &result) {
		float t = (1.0f + textureNoise.noise(p.x * scale)) / 2.0f;
		result = t*c0 + (1.0f - t)*c1;
	}

//	virtual void shade(const SIMDRay &viewer, SIMDHitpoint &p, int idx, rgb& result) {		
//		result = rgb(1,0,0);		
//	}

	// For explicitly sampling the BRDF of the material. vIn and vOut
	// must be normalized.
	virtual void brdf(const Hitpoint &p, const Vector3 &vIn, const Vector3 &vOut, rgb &brdfResult) {
		// BRDF for perfect diffuse is 1/PI:
		float t = (1.0f + textureNoise.noise(p.x * scale)) / 2.0f;
		brdfResult = t*c0 + (1.0f - t)*c1;
		brdfResult /= PI;
	}

	/** 
	* Get a direction to sample further. Direction vIn mus be normalized. The function
	* returns the reflectance for the direction that was sampled. The sampled direction is
	* written into newDirection.
	*
	* This samples a direction for a perfect diffuse material.
	*/
	virtual void sampleDirection(const Hitpoint &p, const Vector3 &vIn, Vector3 &newDirection, rgb &reflectance) {
		static ONB basis;
		float phi = 2.0f * PI * materialSampler.sample();
		float r = sqrtf(materialSampler.sample());
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from normal
		// TODO: save a few instructions and do it here..
		basis.initFromW(p.n);

		// use coordinates in basis:
		newDirection = x * basis.U + y * basis.V + z * basis.W;

		// return probability
		brdf(p, vIn, newDirection, reflectance);
	}

	rgb c0, c1;
	float scale;

protected:
	PerlinNoise textureNoise;

private:
};

#endif