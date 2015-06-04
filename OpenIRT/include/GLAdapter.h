/********************************************************************
	created:	2014/03/04
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	GLAdapter
	file ext:	h
	author:		Tae-Joon Kim (tjkim.kaist@gmail.com)
	
	comment:	Provide OpenGL specific functions for GL based renderer
*********************************************************************/

#pragma once

namespace irt
{
class GLAdapter
{
protected:
	class OpenGLContext;

	OpenGLContext *m_context;
	OpenGLContext *m_contextBackup;
	int m_accum;

public:
	GLAdapter(void);
	virtual ~GLAdapter(void);

	void initGL(int width, int height, void *openGLContext = NULL, void *openGLDC = NULL);

	void resized(int width, int height);

	void beginGL();
	void endGL();
protected:
	bool createOpenGLContext(int width, int height);
};

};