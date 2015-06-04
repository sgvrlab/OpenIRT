#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/glut.h>

#include "GLAdapter.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#include <stdio.h>

using namespace irt;

class GLAdapter::OpenGLContext
{
public:
	OpenGLContext() : context(0), dc(0) {}

	HGLRC context;
	HDC dc;
};

#ifdef _WIN32
HWND g_hWnd = NULL;

LRESULT CALLBACK fakeProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return DefWindowProc(hWnd, msg, wParam, lParam);
};
#endif

GLAdapter::GLAdapter(void) : m_accum(0), m_context(0), m_contextBackup(0)
{
	m_context = new OpenGLContext;
	m_contextBackup = new OpenGLContext;
}

GLAdapter::~GLAdapter(void)
{
	if(m_context->context) wglDeleteContext(m_context->context);

	if(g_hWnd && m_context->dc) ReleaseDC(g_hWnd, m_context->dc);
	if(g_hWnd) DestroyWindow(g_hWnd);

	if(m_context) delete m_context;
	if(m_contextBackup) delete m_contextBackup;
}

void GLAdapter::initGL(int width, int height, void *openGLContext, void *openGLDC)
{
	if(openGLContext && openGLDC)
	{
		m_context->context = (HGLRC)openGLContext;
		m_context->dc = (HDC)openGLDC;
	}
	else createOpenGLContext(width, height);
}

void GLAdapter::resized(int width, int height)
{
#	ifdef _WIN32
	if(g_hWnd) SetWindowPos(g_hWnd, NULL, 0, 0, width, height, 0);
#	endif
}

void GLAdapter::beginGL()
{
	HGLRC context = wglGetCurrentContext();
	HDC dc = wglGetCurrentDC();

	if(context == m_context->context && dc == m_context->dc) 
	{
		m_accum++;
		return;
	}

	m_contextBackup->context = wglGetCurrentContext();
	m_contextBackup->dc = wglGetCurrentDC();

	wglMakeCurrent(NULL, NULL);
	wglMakeCurrent(m_context->dc, m_context->context);
}

void GLAdapter::endGL()
{
	if(m_accum == 0)
	{
		wglMakeCurrent(NULL, NULL);
		wglMakeCurrent(m_contextBackup->dc, m_contextBackup->context);
	}
	else
	{
		m_accum--;
	}
}

bool GLAdapter::createOpenGLContext(int width, int height)
{
#	ifdef _WIN32

	static HINSTANCE hInst = NULL;

	WNDCLASS wc;
	memset(&wc, 0, sizeof(WNDCLASS));
	wc.lpszClassName = "FakeWindow";
	wc.lpfnWndProc = fakeProc;
	wc.hInstance = hInst;
	if(!RegisterClass(&wc))
	{
		printf("Failed: RegisterClass [%d]\n", GetLastError());
		return false;
	}

	if(g_hWnd && m_context->dc) ReleaseDC(g_hWnd, m_context->dc);
	if(g_hWnd) DestroyWindow(g_hWnd);

	g_hWnd = CreateWindow(wc.lpszClassName, wc.lpszClassName, WS_POPUP, 0, 0, width, height, NULL, NULL, hInst, NULL);
	if(!g_hWnd)
	{
		printf("Failed to create fake window [%d].\n", GetLastError());
		return false;
	}

	m_context->dc = GetDC(g_hWnd);

	GLuint pixelFormat;
	static PIXELFORMATDESCRIPTOR pfd =
	{
        sizeof (PIXELFORMATDESCRIPTOR),             
        1,                                          
        PFD_DRAW_TO_WINDOW |                        
        PFD_SUPPORT_OPENGL |
        PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA,                              
		32, 8, 0, 8, 0, 8, 0, 8, 0,
        0, 
		0, 0, 0, 0,                              
        32,                                         
        0,                                          
        0,                                          
        0,                             
        0,                                          
        0, 0, 0                                     
    };

	pixelFormat = ChoosePixelFormat(m_context->dc, &pfd);
	if(!SetPixelFormat(m_context->dc, pixelFormat, &pfd))
	{
		printf("Failed: SetPixelFormat\n");
		return false;
	}

	if(m_context->context) wglDeleteContext(m_context->context);

	if(!(m_context->context = wglCreateContext(m_context->dc)))
	{
		printf("Failed: wglCreateContext\n");
		return false;
	}
#	endif

	return true;
}
