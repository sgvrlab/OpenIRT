#ifndef COMMON_MATERIAL_H
#define COMMON_MATERIAL_H

#include "rgb.h"

#define MATERIAL_DIFFUSE_MAXREFLECTANCE 0.6f

/**
 * Base class for materials.
 * 
 * All derived classes must overwrite the shade() method
 * which generates a color for a Hitpoint.
 */
class Material
{
public:
	Material() {};
	virtual ~Material() {} ;

	/**
	 * Shading functions.
	 */
	virtual void shade(Ray &viewer, Hitpoint &p, rgb& result) = 0;
	//virtual void shade(const SIMDRay &viewer, SIMDHitpoint &p, int idx, rgb& result) = 0;

	/**
	 * For explicitly sampling the BRDF of the material. vIn and vOut
	 * must be normalized.
	 */
	virtual void brdf(const Hitpoint &p, const Vector3 &vIn, const Vector3 &vOut, rgb &brdfResult) {
		brdfResult = rgb(0,0,0);
	}

	/**
	 * Does this material emit light (should be false for almost all material types..) ?
	 */
	virtual inline bool isEmitter() const { return false; }

	/**
	 * Emitted light into a certain direction. Vector direction must be normalized.
	 */	
	virtual void emittedRadiance(const Hitpoint &p, const Vector3 &vIn, rgb& emittedColor) {		
		emittedColor = rgb(0,0,0);
	}

	/**
	* Emitted light into a certain direction. Vector direction must be normalized.
	*/	
	virtual void emittedRadiance(const SIMDHitpoint &p, const Vector3 &vIn, int idx, rgb& emittedColor) {		
		emittedColor = rgb(0,0,0);
	}
	
	/**
	 * Provided for normal (Whitted) ray-tracing.
	 */
	virtual inline bool hasReflection() const { return false; }
	virtual inline bool hasRefraction() const { return false; }
	virtual inline float getReflectance() const { return 0.0f; }
	virtual inline float getOpacity() const { return 1.0f; }

	/** 
	* Get a direction to sample further. Direction vIn mus be normalized. The function
	* returns the reflectance for the direction that was sampled. The sampled direction is
	* written into newDirection.
	*	
	*/
	virtual void sampleDirection(const Hitpoint &p, const Vector3 &vIn, Vector3 &newDirection, rgb &reflectance) {
		reflectance = rgb(0,0,0);
	}
		
protected:

	RandomLinear materialSampler;

private:
};

#endif