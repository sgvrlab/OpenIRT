#include "stdafx.h"
#include "AreaLight.h"
#include "Light.h"
#include "Sample.h"

AreaLight::AreaLight(const Vector3& _pos, const float _width_x, const float _width_z, const int _numGrid, const int typeLight) {
	width_z = _width_z;
	width_x = _width_x;
	numGrid = _numGrid;
	pos = _pos;
	color = rgb(1.0f, 1.0f, 1.0f);
	type = typeLight;
}

void AreaLight::setColor(const rgb& c)
{
	color = c;
}

void AreaLight::generateAreaLight(LightList* lightList) {
	Vector2 posLight;
	Sampler sampler;
	Light genericLight;

	lightList->clear();

	int numApproximation = numGrid*numGrid;
	float posx, posz;
	

	posz = pos.z();
	for (int z = 1; z <= numGrid; ++z) {

		posx = pos.x();
		for (int x = 1; x <= numGrid; ++x) {
			sampler.jitter2(&posLight, 1);

			genericLight.color = rgb(color.r()/(float)numApproximation, color.g()/(float)numApproximation, color.b()/(float)numApproximation);
			genericLight.intensity = 1.f;
			genericLight.type = LIGHT_POINT;

			genericLight.pos.setX( posx + posLight.x() * (width_x / numGrid) );
			genericLight.pos.setY( pos.y() );
			genericLight.pos.setZ( posz + posLight.y() * (width_z / numGrid) );

			// NOTE!
			// Below code has scale problem
			if (type == INSIDE_LIGHT)
				genericLight.cutoff_distance = 5000.0f;
			else if (type == OUTSIDE_LIGHT)
				genericLight.cutoff_distance = FLT_MAX;

			lightList->push_back(genericLight);

			posx += width_x / (float)(numGrid);
		}
		posz += width_z / (float)(numGrid);
	}
}	