/********************************************************************
	created:	2009/04/24
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	Camera
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Camera class, generates rays to shoot into the scene
*********************************************************************/

#pragma once

#include "Vector3.h"
#include "Image.h"
#include "Ray.h"

namespace irt
{

class Camera
{
protected:
	// camera name
	char m_name[256];

	// camera position and orientation
	Vector3 m_eye;
	Vector3 m_center;
	Vector3 m_up;

	// projection properties
	float m_fovy;
	float m_aspect;
	float m_zNear;
	float m_zFar;

	// viewing transformation vectors (normalized)
	Vector3 m_vLookAt;
	Vector3 m_vRight;
	Vector3 m_vUp;

	// for efficiency
	Vector3 m_scaledRight;
	Vector3 m_scaledUp;

	Vector3 m_corner;	// left top corner of image plane on world coordinate space

public:
	Camera(void);
	Camera(
		float eyeX, float eyeY, float eyeZ,
		float centerX, float centerY, float centerZ,
		float upX, float upY, float upZ,
		float fovy, float aspect, float zNear, float zFar
		);
	~Camera(void);

	char *getName() {return m_name;}

	const Vector3& getEye() const {return m_eye;}
	const Vector3& getCenter() const {return m_center;}
	const Vector3& getUp() const {return m_up;}

	const float getFovY() const {return m_fovy;}
	const float getAspect() const {return m_aspect;}
	const float getZNear() const {return m_zNear;}
	const float getZFar() const {return m_zFar;}

	const Vector3& getVLookAt() const {return m_vLookAt;}
	const Vector3& getVRight() const {return m_vRight;}
	const Vector3& getVUp() const {return m_vUp;}
	const Vector3& getScaledRight() const {return m_scaledRight;}
	const Vector3& getScaledUp() const {return m_scaledUp;}
	const Vector3& getCorner() const {return m_corner;}

	void setName(const char *name) {strcpy_s(m_name, 255, name);}

	void setEye(const Vector3& eye) {m_center = (m_center - m_eye) + eye; m_eye = eye; recalculate();}
	void setCenter(const Vector3& center) {m_center = center; recalculate();}
	void setUp(const Vector3& up) {m_up = up; recalculate();}

	void setFovY(float fovy) {m_fovy = fovy; recalculate();}
	void setAspect(float aspect) {m_aspect = aspect; recalculate();}
	void setZNear(float zNear) {m_zNear = zNear; recalculate();}
	void setZFar(float zFar) {m_zFar = zFar; recalculate();}

	inline void getRayWithOrigin(Ray &ray, float x, float y) 
	{
		Vector3 target = m_corner + x*m_scaledRight + y*m_scaledUp;
		ray.setOrigin(m_eye);
		ray.setDirection(unitVector(target-m_eye));	
	}

	inline Vector3 getRayTarget(float x, float y)
	{
		return m_corner + x*m_scaledRight + y*m_scaledUp;
	}

	void recalculate();

	Vector3 getCorner2();

	bool operator == (const Camera &camera)
	{
		if(m_eye != camera.m_eye) return false;
		if(m_center != camera.m_center) return false;
		if(m_up != camera.m_up) return false;
		if(m_fovy != camera.m_fovy) return false;
		if(m_aspect != camera.m_aspect) return false;
		if(m_zNear != camera.m_zNear) return false;
		if(m_zFar != camera.m_zFar) return false;
		return true;
	}

	bool operator != (const Camera &camera)
	{
		return !(*this == camera);
	}
};

};