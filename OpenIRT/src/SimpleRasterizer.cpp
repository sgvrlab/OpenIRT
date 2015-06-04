#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/glut.h>
#include "CommonHeaders.h"
#include "SimpleRasterizer.h"

using namespace irt;

void _drawBB(const AABB &bb, bool fill = false)
{
	if(fill)
	{
#		define BRIGHT_COL glColor3f(1.0f, 1.0f, 1.0f)
#		define DARK_COL glColor3f(0.7f, 0.7f, 0.7f)
		glBegin(GL_QUADS);
		BRIGHT_COL;
		glVertex3f(bb.min.x(), bb.min.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.min.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.min.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.min.x(), bb.min.y(), bb.max.z());
		glEnd();
		glBegin(GL_QUADS);
		BRIGHT_COL;
		glVertex3f(bb.min.x(), bb.max.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.max.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.max.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.min.x(), bb.max.y(), bb.max.z());
		glEnd();
		glBegin(GL_QUADS);
		BRIGHT_COL;
		glVertex3f(bb.min.x(), bb.min.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.min.x(), bb.min.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.min.x(), bb.max.y(), bb.max.z());
		BRIGHT_COL;
		glVertex3f(bb.min.x(), bb.max.y(), bb.min.z());
		glEnd();
		glBegin(GL_QUADS);
		DARK_COL;
		glVertex3f(bb.max.x(), bb.min.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.min.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.max.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.max.y(), bb.min.z());
		glEnd();
		glBegin(GL_QUADS);
		BRIGHT_COL;
		glVertex3f(bb.min.x(), bb.min.y(), bb.min.z());
		BRIGHT_COL;
		glVertex3f(bb.min.x(), bb.max.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.max.y(), bb.min.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.min.y(), bb.min.z());
		glEnd();
		glBegin(GL_QUADS);
		DARK_COL;
		glVertex3f(bb.min.x(), bb.min.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.min.x(), bb.max.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.max.y(), bb.max.z());
		DARK_COL;
		glVertex3f(bb.max.x(), bb.min.y(), bb.max.z());
		glEnd();
	}
	else
	{
		glBegin(GL_LINE_LOOP);
		glVertex3f(bb.min.x(), bb.min.y(), bb.min.z());
		glVertex3f(bb.max.x(), bb.min.y(), bb.min.z());
		glVertex3f(bb.max.x(), bb.min.y(), bb.max.z());
		glVertex3f(bb.min.x(), bb.min.y(), bb.max.z());
		glEnd();
		glBegin(GL_LINE_LOOP);
		glVertex3f(bb.min.x(), bb.max.y(), bb.min.z());
		glVertex3f(bb.max.x(), bb.max.y(), bb.min.z());
		glVertex3f(bb.max.x(), bb.max.y(), bb.max.z());
		glVertex3f(bb.min.x(), bb.max.y(), bb.max.z());
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(bb.min.x(), bb.min.y(), bb.min.z());
		glVertex3f(bb.min.x(), bb.max.y(), bb.min.z());
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(bb.max.x(), bb.min.y(), bb.min.z());
		glVertex3f(bb.max.x(), bb.max.y(), bb.min.z());
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(bb.max.x(), bb.min.y(), bb.max.z());
		glVertex3f(bb.max.x(), bb.max.y(), bb.max.z());
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(bb.min.x(), bb.min.y(), bb.max.z());
		glVertex3f(bb.min.x(), bb.max.y(), bb.max.z());
		glEnd();
	}
}


SimpleRasterizer::SimpleRasterizer(void)
	: m_numModels(0), m_indexList(0), m_numTris(0), m_vertexVBOID(0), m_indexVBOID(0)
{
}

SimpleRasterizer::~SimpleRasterizer(void)
{
	done();
}

void SimpleRasterizer::init(Scene *scene)
{
	Renderer::init(scene);

	if(!scene) return;

	GLenum err;
	beginGL();
	err = glewInit();
	endGL();

	if(err)
	{
		initGL(m_width, m_height);
		beginGL();
		err = glewInit();
		endGL();
		if(err)
		{
			printf("Failed: SimpleRasterizer::init\n");
			return;
		}
	}

	done();

	m_intersectionStream = scene->getIntersectionStream();

	beginGL();

	sceneChanged();

	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);							
	glEnable(GL_DEPTH_TEST);					
	glDepthFunc(GL_LEQUAL);	
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	m_numModels = scene->getNumModels();
	err = glGetError();

	m_indexList = new unsigned int*[m_numModels];

	m_numTris = new int[m_numModels];

	m_vertexVBOID = new GLuint[m_numModels];
	m_indexVBOID = new GLuint[m_numModels];
	
	char GLVersion[100];
	const char *ver = (const char*)glGetString(GL_VERSION);
	strcpy_s(GLVersion, ver);
	printf("GL version = %s\n", GLVersion);

	// convert version string to integer : 1.0.0 -> 100000
	int len = (int)strlen((const char*)GLVersion);
	for(int i=0;i<len;i++)
	{
		if(GLVersion[i] == '.')
			GLVersion[i] = '0';
	}
	GLVersion[len] = '0';
	GLVersion[++len] = 0;

	m_GLVersion = atoi(GLVersion);

	if(m_GLVersion >= 105000)
	{
		glGenBuffers(m_numModels, m_vertexVBOID);
		glGenBuffers(m_numModels, m_indexVBOID);
	}

	for(int i=0;i<m_numModels;i++)
	{
		Model *model = scene->getModelList()[i];
		int numVerts = model->getNumVertexs();
		int numTris = model->getNumTriangles();
		m_indexList[i] = new unsigned int[numTris*3];

		for(int j=0;j<numTris;j++)
		{
			Triangle *tri = model->getTriangle(j);
			m_indexList[i][j*3+0] = tri->p[0];
			m_indexList[i][j*3+1] = tri->p[1];
			m_indexList[i][j*3+2] = tri->p[2];
		}

		if(m_GLVersion >= 105000)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBOID[i]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*numVerts, model->getVertex(0), GL_STATIC_DRAW);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexVBOID[i]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*numTris*3, m_indexList[i], GL_STATIC_DRAW);
		}

		m_numTris[i] = numTris;
	}
	/*
	for(int i=0;i<m_numModels;i++)
	{
		Model *model = (Model*)scene->getModelList()[i];
		int numVerts = model->getNumVertexs();
		int numTris = model->getNumTriangles();

		if(m_GLVersion >= 105000)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBOID[i]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(SimpVertex)*(__int64)numVerts, model->getVertex(0), GL_STATIC_DRAW);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*(__int64)numVerts, model->getVertex(0), GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexVBOID[i]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(SimpTriangle)*(__int64)numTris, model->getTriangle(0), GL_STATIC_DRAW);
		}

		m_numTris[i] = numTris;
	}
	*/
	endGL();
}

void SimpleRasterizer::done()
{
	if(m_indexList)
	{
		for(int i=0;i<m_numModels;i++)
			delete[] m_indexList[i];
		delete[] m_indexList;
		m_indexList = NULL;
	}
	if(m_numTris)
	{
		delete[] m_numTris;
		m_numTris = NULL;
	}

	beginGL();
	if(m_vertexVBOID)
	{
		if(m_GLVersion >= 105000) glDeleteBuffers(m_numModels, m_vertexVBOID);
		delete[] m_vertexVBOID;
		m_vertexVBOID = NULL;
	}
	if(m_indexVBOID)
	{
		if(m_GLVersion >= 105000) glDeleteBuffers(m_numModels, m_indexVBOID);
		delete[] m_indexVBOID;
		m_indexVBOID = NULL;
	}
	endGL();
}

void SimpleRasterizer::resized(int width, int height)
{
	Renderer::resized(width, height);

	GLAdapter::resized(width, height);

	beginGL();
	glViewport(0, 0, width, height);
	endGL();
}

void SimpleRasterizer::sceneChanged()
{
	beginGL();
	Emitter emitter = m_scene->getEmitter(0);
	const RGBf &ambient = emitter.color_Ka;
	const RGBf &diffuse = emitter.color_Kd;
	GLfloat lightAmbient[] = {ambient.r(), ambient.g(), ambient.b(), 1.0f};
	GLfloat lightDiffuse[] = {diffuse.r(), diffuse.g(), diffuse.b(), 1.0f};
	GLfloat lightPosition[] = {emitter.pos.x(), emitter.pos.y(), emitter.pos.z(), 0.0f};
	if(emitter.type == Emitter::PARALLELOGRAM_LIGHT)
	{
		Vector3 pos = emitter.planar.corner + 0.5f * emitter.planar.v1 + 0.5f * emitter.planar.v2;
		lightPosition[0] = pos.x();
		lightPosition[1] = pos.y();
		lightPosition[2] = pos.z();
	}
	glLightfv(GL_LIGHT1, GL_AMBIENT, lightAmbient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, lightDiffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, lightPosition);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHTING);
	endGL();
}

void SimpleRasterizer::render(Camera *camera, Image *image, unsigned int seed)
{
	beginGL();
	sceneChanged();

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluPerspective(camera->getFovY(), camera->getAspect(), camera->getZNear(), camera->getZFar());

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	gluLookAt(
		camera->getEye().x(), camera->getEye().y(), camera->getEye().z(),
		camera->getCenter().x(), camera->getCenter().y(), camera->getCenter().z(), 
		camera->getUp().x(), camera->getUp().y(), camera->getUp().z());

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);							

#	define BUFFER_OFFSET(i) ((char *)NULL + (i))

	render(&m_scene->getSceneGraph());

	if(image)
	{
		if(image->bpp == 4)
			glReadPixels(0, 0, image->width, image->height, GL_RGBA, GL_UNSIGNED_BYTE, image->data);
		else
			glReadPixels(0, 0, image->width, image->height, GL_RGB, GL_UNSIGNED_BYTE, image->data);
	}
	endGL();
}

void _drawBB(Model *model, BVHNode *node, int depth)
{
	glDisable(GL_LIGHTING);
	glColor3f(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
	_drawBB(AABB(node->min, node->max));

	//if(depth == 4) return;

	if((node->left & 3) == 3)
	{
		//Triangle *tri = model->getTriangle(node->right);
		//const Vector3 &v0 = model->getVertex(tri->p[0])->v;
		//const Vector3 &v1 = model->getVertex(tri->p[1])->v;
		//const Vector3 &v2 = model->getVertex(tri->p[2])->v;
		//glDisable(GL_LIGHTING);
		//glColor3f(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
		//glBegin(GL_TRIANGLES);
		//glVertex3f(v0.e[0], v0.e[1], v0.e[2]);
		//glVertex3f(v1.e[0], v1.e[1], v1.e[2]);
		//glVertex3f(v2.e[0], v2.e[1], v2.e[2]);
		//glEnd();
		return;
	}

	_drawBB(model, model->getBV(node->left >> 2), depth+1);
	_drawBB(model, model->getBV(node->right >> 2), depth+1);
}

void SimpleRasterizer::render(SceneNode *sceneNode)
{
	if(sceneNode->model)
	{
		glPushMatrix();

		_drawBB(sceneNode->model, sceneNode->model->getBV(0), 0);
#		if 0

		Matrix mat = sceneNode->matrix;
		mat.transpose();
		glMultMatrixf((GLfloat*)mat.x);
		Model *model = sceneNode->model;
		int modelIndex = m_scene->getModelListMap()[model];
		if(m_GLVersion >= 105000)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBOID[modelIndex]);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, sizeof(Vertex), BUFFER_OFFSET(0));
			//glVertexPointer(3, GL_FLOAT, sizeof(SimpVertex), BUFFER_OFFSET(0));
			
			glEnableClientState(GL_NORMAL_ARRAY);
			glNormalPointer(GL_FLOAT, sizeof(Vertex), BUFFER_OFFSET(16));
			/*
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_FLOAT, sizeof(Vertex), BUFFER_OFFSET(32));
			*/
		
			//glClientActiveTexture(GL_TEXTURE0);
			//glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			//glTexCoordPointer(2, GL_FLOAT, sizeof(Vertex), BUFFER_OFFSET(48));
 
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexVBOID[modelIndex]);

			glDrawElements(GL_TRIANGLES, m_numTris[modelIndex]*3, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
			GLenum err = glGetError();

			glDisableClientState(GL_VERTEX_ARRAY);

			/*
			/////////////////////////////////////////////////////
			//glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBOID[modelIndex]);
			//glEnableClientState(GL_VERTEX_ARRAY);
			
			Vertex *v = model->getVertex(0);
			unsigned int *idx = m_indexList[0];

			//glBindBuffer(GL_ARRAY_BUFFER, m_vertexVBOID[modelIndex]);
			glVertexPointer(3, GL_FLOAT, sizeof(Vertex), v);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexVBOID[modelIndex]);
			glDrawElements(GL_TRIANGLES, m_numTris[modelIndex]*3, GL_UNSIGNED_INT, 0);
			glDisableClientState(GL_VERTEX_ARRAY);
			/////////////////////////////////////////////////////
			*/
		}
		else
		{
			GLfloat mat[4] = {1.0f, 1.0f, 1.0f, 1.0f};
#			define SET_MATERIAL(m) memcpy(mat, m, sizeof(float)*3)
			for(int j=0;j<m_numTris[modelIndex];j++)
			{
				Triangle *tri = model->getTriangle(j);
				SET_MATERIAL(model->getMaterial(tri->material).getMatKa().e);
				glMaterialfv(GL_FRONT, GL_AMBIENT, mat);
				SET_MATERIAL(model->getMaterial(tri->material).getMatKd().e);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, mat);
				glBegin(GL_TRIANGLES);
				glNormal3f(model->getVertex(tri->p[0])->n.x(), model->getVertex(tri->p[0])->n.y(), model->getVertex(tri->p[0])->n.z());
				glVertex3f(model->getVertex(tri->p[0])->v.x(), model->getVertex(tri->p[0])->v.y(), model->getVertex(tri->p[0])->v.z());
				glNormal3f(model->getVertex(tri->p[1])->n.x(), model->getVertex(tri->p[1])->n.y(), model->getVertex(tri->p[1])->n.z());
				glVertex3f(model->getVertex(tri->p[1])->v.x(), model->getVertex(tri->p[1])->v.y(), model->getVertex(tri->p[1])->v.z());
				glNormal3f(model->getVertex(tri->p[2])->n.x(), model->getVertex(tri->p[2])->n.y(), model->getVertex(tri->p[2])->n.z());
				glVertex3f(model->getVertex(tri->p[2])->v.x(), model->getVertex(tri->p[2])->v.y(), model->getVertex(tri->p[2])->v.z());
				glEnd();
			}
		}

#		endif

		glPopMatrix();
	}

	if(sceneNode->hasChilds())
	{
		for(size_t i=0;i<sceneNode->childs->size();i++)
		{
			render(sceneNode->childs->at(i));
		}
	}
}