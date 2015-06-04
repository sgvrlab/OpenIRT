#include "StdAfx.h"
#include "OpenGLDevice.h"
#include <GL/glut.h>

OpenGLDevice::OpenGLDevice(void)
{
	m_RenderContext = NULL;
	m_DeviceContext = NULL;
}

OpenGLDevice::~OpenGLDevice(void)
{
	destroy();
}

OpenGLDevice::OpenGLDevice(HWND& window,int stencil)
{
	create(window,stencil);
}

OpenGLDevice::OpenGLDevice(HDC& deviceContext,int stencil)
{
	create(deviceContext,stencil);
}

bool OpenGLDevice::create(HWND& window,int stencil)
{
	HDC deviceContext = ::GetDC(window);
	
	if (!create(deviceContext,stencil))
	{
		::ReleaseDC(window, deviceContext);

		return false;
	}

	::ReleaseDC(window, deviceContext);
	
	return true;
}

bool OpenGLDevice::create(HDC& deviceContext,int stencil)
{
	if (!deviceContext)
	{
		return false;
	}

	if (!setDCPixelFormat(deviceContext,stencil))
	{
		return false;
	}

	m_RenderContext = wglCreateContext(deviceContext);
	wglMakeCurrent(deviceContext, m_RenderContext);

	OpenGLDevice::m_DeviceContext = deviceContext;
	
	return true;
}

void OpenGLDevice::destroy()
{
	if (m_RenderContext != NULL)
	{
		wglMakeCurrent(NULL,NULL);
		wglDeleteContext(m_RenderContext);
	}
}

void OpenGLDevice::makeCurrent(bool disableOther)
{
	if (m_RenderContext != NULL)
	{
			//should all other device contexts
			//be disabled then?
			if (disableOther) 
				wglMakeCurrent(NULL,NULL);

			wglMakeCurrent(m_DeviceContext, m_RenderContext);
	}
}

bool OpenGLDevice::setDCPixelFormat(HDC& deviceContext,int stencil)
{
	int pixelFormat;
	DEVMODE resolution;

	EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &resolution);
	
	static PIXELFORMATDESCRIPTOR pixelFormatDesc =
	{
        sizeof (PIXELFORMATDESCRIPTOR),             
        1,                                          
        PFD_DRAW_TO_WINDOW |                        
        PFD_SUPPORT_OPENGL |
        PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA,                              
        (BYTE)resolution.dmBitsPerPel,                                         
        0, 0, 0, 0, 0, 0,                           
        0, 
		0,                                       
        0, 
		0, 0, 0, 0,                              
        16,                                         
        stencil,                                          
        0,                                          
        0,                             
        0,                                          
        0, 0, 0                                     
    };

    
    
    pixelFormat = ChoosePixelFormat (deviceContext,
									&pixelFormatDesc);

    if (!SetPixelFormat(deviceContext, pixelFormat, 
		&pixelFormatDesc)) 
	{
      return false ;
    }

    return true;
}