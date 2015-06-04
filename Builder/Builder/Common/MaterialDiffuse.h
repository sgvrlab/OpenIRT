#ifndef COMMON_MATERIALDIFFUSE_H
#define COMMON_MATERIALDIFFUSE_H

#include "Materials.h"

/**
* Diffuse material, color only
*/
class MaterialDiffuse : public Material
{
public:
	MaterialDiffuse(rgb &color) { 		
		setColor(color);
	};

	MaterialDiffuse(MaterialDiffuse &other) {
		setColor(other.c);
	}

	MaterialDiffuse() { 
		setColor(rgb(0,0,0));
	};

	~MaterialDiffuse() {};

	void shade(Ray &viewer, Hitpoint &p, rgb &result) {
		result = c;
	}

	//virtual void shade(const SIMDRay &viewer, SIMDHitpoint &p, int idx, rgb& result) {
	//	result = c;
	//}

	// sets the constant color of this material
	void setColor(rgb &newColor) {
		c = newColor;

		// BRDF for perfect diffuse is 1/PI:
		staticBRDF = c / PI;
	}

	// sungeui start -----------
	rgb getColor (void) {
		return c;	
	}
	// sungeui end --------------

	// For explicitly sampling the BRDF of the material. vIn and vOut
	// must be normalized.
	virtual void brdf(const Hitpoint &p, const Vector3 &vIn, const Vector3 &vOut, rgb &brdfResult) {
		brdfResult = staticBRDF;
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

		// return probability (local color * parameter)
		reflectance = c*MATERIAL_DIFFUSE_MAXREFLECTANCE;		
	}
	
protected:
	rgb staticBRDF; 
	rgb c;	

private:
};

#endif