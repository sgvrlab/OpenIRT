#ifndef COMMON_MATERIALSPECULARANDBITMAP_H
#define COMMON_MATERIALSPECULARANDBITMAP_H

#include "Materials.h"

/**
* Specular material, color only
*/
class MaterialSpecularAndBitmap : public Material
{
public:
	MaterialSpecularAndBitmap(const char *textureFileName) { 
		setRefractionIndex(1.33f);		
		calcProbabilities();

		TextureManager *texMan = TextureManager::getSingletonPtr();

		texPtr = texMan->loadTexture(textureFileName);

		if (!texPtr) {
			LogManager *log = LogManager::getSingletonPtr();						
			log->logMessage(LOG_ERROR, texMan->getLastErrorString());
		}
	};

	MaterialSpecularAndBitmap() { 
		setRefractionIndex(1.33f);
		setColor(rgb(0,0,0));
		calcProbabilities();
	};

	~MaterialSpecularAndBitmap() {
		TextureManager *texMan = TextureManager::getSingletonPtr();

		if (texPtr) {
			texMan->unloadTexture(texPtr);
		}
	};

	/*
	void shade(Vector3 &viewer, Hitpoint &p, rgb &result);

	virtual void shade(Vector3 &viewer, SIMDHitpoint &p, int idx, rgb& result);
	*/
	void shade(Vector3 &viewer, Hitpoint &p, rgb &result) {}
	void shade(Vector3 &viewer, SIMDHitpoint &p, int idx, rgb& result) {}
	

	// sets the constant color of this material
	void setColor(rgb &newColor) {
		c = newColor;
	}

	// For explicitly sampling the BRDF of the material. vIn and vOut
	// must be normalized.
	//
	// This is the dirac delta function that is only != 0 for the direction
	// that is exactly the mirror direction.
	virtual void brdf(const Hitpoint &p, const Vector3 &vIn, const Vector3 &vOut, rgb &brdfResult) {
		Vector3 reflectedDirection = reflect(vIn, p.n);
		brdfResult = (reflectedDirection == vOut)?rgb(specularReflectance, specularReflectance, specularReflectance):rgb(0,0,0);
	}

	/** 
	* Get a direction to sample further. Direction vIn mus be normalized. The function
	* returns the reflectance for the direction that was sampled. The sampled direction is
	* written into newDirection.
	*
	* This samples the mirror direction OR the refraction direction randomly
	*/
	virtual void sampleDirection(const Hitpoint &p, const Vector3 &vIn, Vector3 &newDirection, rgb &reflectance) {
		float r = materialSampler.sample();

		// Choose which reflection to take:
		//
		if (r < probabilityReflection) { // reflection
			newDirection = reflect(vIn, p.n);
			reflectance = rgb(probabilityReflection, probabilityReflection, probabilityReflection);		
		}
		else if (r < probabilityReflection + probabilityRefraction) { // refraction
			float dn = dot(vIn, p.n);

			// Air -> Material
			if (dn < 0.0f) {				
				dn = -dn;
				float root = 1.0f - (indexInv*indexInv) * (1.0f - dn*dn);
				newDirection = vIn * indexInv + p.n * (indexInv*dn - sqrt(root));
				reflectance = rgb(probabilityRefraction, probabilityRefraction, probabilityRefraction);
			}
			else {
				// Material -> Air
				float root = 1.0f - (refractionIndex*refractionIndex) * (1.0f - dn*dn);

				if (root >= 0.0f) {	
					newDirection = vIn*refractionIndex - p.n*(refractionIndex * dn - sqrt(root));
					reflectance = rgb(probabilityRefraction, probabilityRefraction, probabilityRefraction);
				}
				else {
					// if root < 0.0f, then we have total internal reflection 
					// and therefore do not need to trace the refraction..
					newDirection = Vector3(0,0,0);
					reflectance = rgb(0,0,0);
				}
			}			
		}
		else if (r < probabilityReflection + probabilityRefraction + probabilityDiffuse) { // diffuse reflection			
			reflectance = rgb(0,0,0);
		}
		else { // absorption
			reflectance = rgb(0,0,0);
		}


	}

	/**
	* Provided for normal (Whitted) ray-tracing.
	*/
	virtual inline bool hasReflection() const { return specularReflectance > 0.0f; }
	virtual inline bool hasRefraction() const { return opacity < 1.0f; }
	virtual inline float getReflectance() const { return specularReflectance; }
	virtual inline float getOpacity() const { return opacity; }

	void setReflectance(float newReflectance) {
		specularReflectance = newReflectance;
		calcProbabilities();
	}

	void setOpacity(float newOpacity) {
		opacity = newOpacity;
		calcProbabilities();
	}

	void setRefractionIndex(float newIndex) {
		refractionIndex = newIndex;
		R0 = (refractionIndex - 1.0f) / (refractionIndex + 1.0f);
		R0 *= R0;
		indexInv = 1.0f / refractionIndex;
	}

	//protected:
	// for the time being.

	// calculate probability that a ray will be refracted, specularly reflected, diffusely reflected or absorbed
	inline void calcProbabilities() {
		if (specularReflectance > 0.0f && opacity < 1.0f) {// refractive AND reflective:
			probabilityReflection = specularReflectance * 0.5f;
			probabilityRefraction = (1.0f - opacity) * 0.5f;
			probabilityDiffuse = (1.0f - probabilityReflection - probabilityRefraction) * MATERIAL_DIFFUSE_MAXREFLECTANCE;
		}
		else { // either reflective or refractive:
			probabilityReflection = specularReflectance;
			probabilityRefraction = 1.0f - opacity;
			probabilityDiffuse = (1.0f - probabilityReflection - probabilityRefraction) * MATERIAL_DIFFUSE_MAXREFLECTANCE;
		}
	}

	rgb c;
	float specularReflectance; 
	float opacity;
	float refractionIndex;

	float probabilityReflection;
	float probabilityRefraction;
	float probabilityDiffuse;

	// precalc vars for refraction
	float R0;
	float indexInv;

	// bitmap we're using for shading
	BitmapTexture *texPtr;

private:
};

#endif