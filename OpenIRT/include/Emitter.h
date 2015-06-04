#pragma once

#include <vector>
#include <string>
/*
#include "Material.h"

class Emitter : public Material
{
public:
	Vector3 pos;
};
*/

#include "Parallelogram.h"

namespace irt
{

class Emitter
{
public:
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

	Emitter()
	{ 
		clear();
		color_Kd = RGBf(1.0f, 1.0f, 1.0f);	// default material
		intensity = 1.0f;
		type = POINT_LIGHT;
	};

	~Emitter() {};

	void clear()
	{
		color_Ka = RGBf(0.0f, 0.0f, 0.0f);
		color_Kd = RGBf(0.0f, 0.0f, 0.0f);
		color_Ks = RGBf(0.0f, 0.0f, 0.0f);

		pos = Vector3(0.0f, 0.0f, 0.0f);
		planar.corner = Vector3(0.0f, 0.0f, 0.0f);
		planar.v1 = Vector3(0.0f, 0.0f, 0.0f);
		planar.v2 = Vector3(0.0f, 0.0f, 0.0f);
		planar.normal = Vector3(0.0f, 0.0f, 0.0f);
		spotTarget.corner = Vector3(0.0f, 0.0f, 0.0f);
		spotTarget.v1 = Vector3(0.0f, 0.0f, 0.0f);
		spotTarget.v2 = Vector3(0.0f, 0.0f, 0.0f);
		spotTarget.normal = Vector3(0.0f, 0.0f, 0.0f);
		numScatteringPhotons = 0;
		intensity = 0.0f;

		sprintf_s(name, 60, "emitter");
		environmentTexName[0] = 0;
		isCosLight = false;
		targetType = LIGHT_TARGET_NONE;
	}

	RGBf color_Ka;		// ambient light
	RGBf color_Kd;		// diffuse light
	RGBf color_Ks;		// specular light
	Type type;			// light type
	TargetType targetType;	// light target type

	char name[60];

	Vector3 pos;			// position for point light
	Parallelogram planar;
	Parallelogram spotTarget;
	int numScatteringPhotons;
	float intensity;
	bool isCosLight;

	char environmentTexName[256];	// file name of environment cube map

	void setName(const char *name)
	{
		strcpy_s(this->name, 60, name);
	}

	Type setType(const char *typeName)
	{
		if(strcmp(typeName, "ParallelogramLight") == 0)
			type = PARALLELOGRAM_LIGHT;
		else if(strcmp(typeName, "EnvironmentLight") == 0)
			type = ENVIRONMENT_LIGHT;
		else
			type = POINT_LIGHT;
		return type;
	}

	void getTypeName(char *typeName, int size)
	{
		switch(type)
		{
		case PARALLELOGRAM_LIGHT : sprintf_s(typeName, size, "ParallelogramLight"); break;
		case TRIANGULAR_LIGHT : sprintf_s(typeName, size, "TriangularLight"); break;
		case ENVIRONMENT_LIGHT : sprintf_s(typeName, size, "EnvironmentLight"); break;
		case POINT_LIGHT : sprintf_s(typeName, size, "PointLight"); break;
		}
	}

	bool hasSpot() {return spotTarget.corner.x() != FLT_MAX;}

	inline Vector3 sample(unsigned int prevRnd) const
	{
		if(type == Emitter::PARALLELOGRAM_LIGHT)
			return planar.sample(prevRnd);
		else
			return pos;
	}

	inline Vector3 sampleEmitDirection(const Vector3 &ori, unsigned int prevRnd) const
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
		return ori;
	}
};

typedef std::vector<Emitter> EmitterList;

};