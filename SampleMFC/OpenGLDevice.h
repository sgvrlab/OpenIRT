#pragma once
class OpenGLDevice
{
public:
	OpenGLDevice(HDC& deviceContext,int stencil = 0);
	OpenGLDevice(HWND& window,int stencil = 0);
	OpenGLDevice(void);

	bool create(HDC& deviceContext,int  stencil = 0);
	bool create(HWND& window,int stencil = 0);

	void destroy();
	void makeCurrent(bool disableOther = true);

	~OpenGLDevice(void);

	HGLRC getRenderContext() {return m_RenderContext;}
	HDC getDeviceContext() {return m_DeviceContext;}

protected:
	bool setDCPixelFormat(HDC& deviceContext,int stencil);
	
	HGLRC m_RenderContext;
	HDC m_DeviceContext;
};

