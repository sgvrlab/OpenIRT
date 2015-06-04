#include "defines.h"
#include "CommonHeaders.h"
#include "handler.h"

#include "Camera.h"

#include <math.h>

using namespace irt;

Camera::Camera(void)
{
}

Camera::Camera(
	   float eyeX, float eyeY, float eyeZ,
	   float centerX, float centerY, float centerZ,
	   float upX, float upY, float upZ,
	   float fovy, float aspect, float zNear, float zFar
	   ) :
m_eye(Vector3(eyeX, eyeY, eyeZ)),
m_center(Vector3(centerX, centerY, centerZ)),
m_up(Vector3(upX, upY, upZ)),
m_fovy(fovy), m_aspect(aspect), m_zNear(zNear), m_zFar(zFar)
{
	recalculate();
}


Camera::~Camera(void)
{
}

void Camera::recalculate()
{
	m_vLookAt = m_center - m_eye;
	m_vLookAt.makeUnitVector();

	m_vRight = cross(m_vLookAt, m_up);
	m_vRight.makeUnitVector();

	m_vUp = -cross(m_vRight, m_vLookAt);

	float halfHeight = m_zNear*tan(m_fovy*PI/360);
	float halfWidth = m_aspect * halfHeight;

	m_scaledRight = halfWidth*m_vRight*2;
	m_scaledUp = halfHeight*m_vUp*2;

	m_corner = m_eye - (halfWidth*m_vRight) - (halfHeight*m_vUp) + (m_zNear*m_vLookAt);
}

Vector3 Camera::getCorner2()
{
	float halfHeight = m_zNear*tan(m_fovy*PI/360);
	float halfWidth = m_aspect * halfHeight;

	return m_eye - (halfWidth*m_vRight) - (halfHeight*m_vUp) + (m_zNear*m_vLookAt);
}