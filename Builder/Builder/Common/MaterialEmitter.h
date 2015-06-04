#ifndef COMMON_MATERIALEMITTER_H
#define COMMON_MATERIALEMITTER_H

#include "Materials.h"

/**
* Emitter material, color only
*/
class MaterialEmitter : public Material
{
public:
	MaterialEmitter(rgb &color) { 
		setColor(color);
	};

	MaterialEmitter() { 
		setColor(rgb(0,0,0));
	};

	~MaterialEmitter() {};

	void shade(Ray &viewer, Hitpoint &p, rgb &result) {
		result = c*50.0f;
	}

//	virtual void shade(const SIMDRay &viewer, SIMDHitpoint &p, int idx, rgb& result) {
//		result = c*50.0f;
//	}

	/**
	 * Does this material emit light (should be false for almost all material types..) ?
	 */
	virtual inline bool isEmitter() const { return true; }

	/**
	* Emitted light into a certain direction. Vector direction must be normalized.
	*/	
	virtual void emittedRadiance(const Hitpoint &p, const Vector3 &vIn, rgb& emittedColor) {		
		emittedColor = c*100.0f;
	}

	/**
	* Emitted light into a certain direction. Vector direction must be normalized.
	*/	
	virtual void emittedRadiance(const SIMDHitpoint &p, const Vector3 &vIn, int idx, rgb& emittedColor) {		
		emittedColor = c*100.0f;
	}

	// sets the constant color of this material
	void setColor(rgb &newColor) {
		c = newColor;
	}


protected:
	rgb c;

private:
};

#endif