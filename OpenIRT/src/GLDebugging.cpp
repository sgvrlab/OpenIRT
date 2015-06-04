#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/glut.h>
#include "CommonHeaders.h"
#include "GLDebugging.h"
#include "random.h"
#include <io.h>
#include "defines.h"
#include "handler.h"
#include "OpenIRT.h"
#include "Model.h"
#include "HCCMesh2.h"
#include "FrustumCulling.h"
#include "OOCVoxelManager.h"
#include "CommonOptions.h"

#define USE_LIGHTING
#define TEST_MODEL_TYPE Model

using namespace irt;

void drawBB(const AABB &bb, bool fill = false)
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

GLDebugging::GLDebugging(void)
	: m_photons(0), m_numPhotons(0), m_octreeList(0)
{
}

GLDebugging::~GLDebugging(void)
{
	if(m_photons) delete[] m_photons;
}

void GLDebugging::init(Scene *scene)
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
			printf("Failed: GLDebugging::init\n");
			return;
		}
	}

	done();

	m_intersectionStream = scene->getIntersectionStream();

	sceneChanged();

	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);							
	glEnable(GL_DEPTH_TEST);					
	glDepthFunc(GL_LEQUAL);	
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void GLDebugging::done()
{
}

void GLDebugging::resized(int width, int height)
{
	Renderer::resized(width, height);

	glViewport(0, 0, width, height);
}

void GLDebugging::sceneChanged()
{
#	ifdef USE_LIGHTING
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
#	endif
	return ;
#	if 1	// use upper ASVO only
	char fileNameVoxel[MAX_PATH];
	sprintf_s(fileNameVoxel, MAX_PATH, "%s_voxel.ooc", m_scene->getASVOFileBase());
	m_octree.load(fileNameVoxel, false);
#	else
	typedef struct OOCVoxel_t
	{
		int rootChildIndex;
		int startDepth;
		int offset;
		int numVoxels;
		AABB rootBB;
	} OOCVoxel;

	FILE *fpHigh, *fpLow, *fpHeader, *fp;
	char fileName[MAX_PATH];

	sprintf(fileName, "%s_fullVoxel.ooc", m_scene->getASVOFileBase());
	fopen_s(&fp, fileName, "rb");
	if(!fp)
	{
		fopen_s(&fp, fileName, "wb");
		sprintf(fileName, "%s_voxel.ooc", m_scene->getASVOFileBase());
		fopen_s(&fpHigh, fileName, "rb");
		sprintf(fileName, "%s_OOCVoxel.ooc", m_scene->getASVOFileBase());
		fopen_s(&fpLow, fileName, "rb");
		sprintf(fileName, "%s_OOCVoxel.hdr", m_scene->getASVOFileBase());
		fopen_s(&fpHeader, fileName, "rb");

		__int64 numHighVoxels = (_filelengthi64(_fileno(fpHigh)) - sizeof(OctreeHeader)) / sizeof(Voxel);
		__int64 numLowVoxels = _filelengthi64(_fileno(fpLow)) / sizeof(Voxel);
		__int64 numOVoxels = _filelengthi64(_fileno(fpHeader)) / sizeof(OOCVoxel);

		Voxel *octree;
		printf("Allocating %I64d bytes...\n", (numHighVoxels + numLowVoxels)*sizeof(Voxel));
		if(!(octree = new Voxel[numHighVoxels + numLowVoxels]))
		{
			printf("Memory allocating error! (%I64d bytes required)\n", (numHighVoxels + numLowVoxels)*sizeof(Voxel));
			fclose(fpHigh);
			fclose(fpLow);
			fclose(fpHeader);
			return;
		}

		m_octree.setOctreePtr(octree);
		printf("Read voxels...\n");
		// read OOC voxel header
		OOCVoxel *oocVoxelList = new OOCVoxel[numOVoxels];
		fread(oocVoxelList, sizeof(OOCVoxel), numOVoxels, fpHeader);
		fclose(fpHeader);
	
		__int64 offset = 0;
		// read high level voxels
		fread(&m_octree.getHeader(), sizeof(OctreeHeader), 1, fpHigh);
		offset += fread(&m_octree[offset], sizeof(Voxel), numHighVoxels, fpHigh);
		fclose(fpHigh);

		// read low level voxels
		fread(&m_octree[offset], sizeof(Voxel), numLowVoxels, fpLow);
		fclose(fpLow);

		printf("Link voxels...\n");
		// link all
		for(int i=0;i<numOVoxels;i++)
		{
			const OOCVoxel &oocVoxel = oocVoxelList[i];

			m_octree[oocVoxel.rootChildIndex].setChildIndex((int)(offset/8));
			//m_octree[oocVoxel.rootChildIndex].setLink2Low(i);
			for(int j=0;j<oocVoxel.numVoxels;j++)
			{
				Voxel &voxel = m_octree[offset+j];
				if(voxel.hasChild())
					voxel.setChildIndex(voxel.getChildIndex() + (int)(offset/8));
			}

			offset += oocVoxel.numVoxels;
		}

		fwrite(&m_octree.getHeader(), sizeof(OctreeHeader), 1, fp);
		fwrite(octree, sizeof(Voxel), numHighVoxels + numLowVoxels, fp);
	}
	else
	{
		__int64 numFullVoxels = (_filelengthi64(_fileno(fp)) - sizeof(OctreeHeader)) / sizeof(Voxel);

		Voxel *octree;
		printf("Allocating %I64d bytes...\n", numFullVoxels*sizeof(Voxel));
		if(!(octree = new Voxel[numFullVoxels]))
		{
			printf("Memory allocating error! (%I64d bytes required)\n", numFullVoxels*sizeof(Voxel));
			fclose(fp);
			return;
		}

		printf("Read voxels...\n");
		fread(&m_octree.getHeader(), sizeof(OctreeHeader), 1, fp);
		fread(octree, sizeof(Voxel), numFullVoxels, fp);
		m_octree.setOctreePtr(octree);
	}

#	endif

	//tracePhotons();
	//buildPhotonKDTree();

	//m_octree.makeLOD(0);

	m_bbList.clear();

#	ifdef VIS_PHOTONS
	char fileNamePhotons[MAX_PATH];
	sprintf_s(fileNamePhotons, MAX_PATH, "%s_photonKDTree.ooc", m_scene->getASVOFileBase());

	FILE *fp;
	fopen_s(&fp, fileNamePhotons, "rb");

	fread(&m_numPhotons, sizeof(int), 1, fp);

	if(m_photons)
		delete[] m_photons;

	m_photons = new Photon[m_numPhotons];
	fread(m_photons, sizeof(Photon), m_numPhotons, fp);

	fclose(fp);

#	endif
}

void GLDebugging::render(Camera *camera, Image *image, unsigned int seed)
{
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

	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);							
	glEnable(GL_DEPTH_TEST);					
	glDepthFunc(GL_LEQUAL);	
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glEnable (GL_LINE_SMOOTH);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint (GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(2.0f);

	glLineWidth(1.0f);
	glPointSize(3.0f);
	glDisable(GL_LIGHTING);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	//drawBB(AABB(Vector3(-88.0f, 1140.0f, -252.0f), Vector3(-84.0f, 1144.0f, -248.0f)));
	//glColor3f(1.0f, 0.0f, 0.0f);
	//glBegin(GL_POLYGON);
	//glVertex3f(-87.999969f, 1144.0000f, -249.16913f);
	//glVertex3f(-87.999969f, 1143.3025f, -248.00000f);
	//glVertex3f(-85.353424f, 1140.0000f, -248.00000f);
	//glVertex3f(-84.000000f, 1140.0000f, -250.83081f);
	//glVertex3f(-84.000000f, 1140.6975f, -251.99998f);
	//glVertex3f(-86.646523f, 1144.0000f, -251.99998f);
	//glEnd();

	//glVertex3f(-86.646523f, 1144.0000f, -251.99998f);
	//glVertex3f(-85.353424f, 1140.0000f, -248.00000f);
	//glVertex3f(-87.999969f, 1143.3025f, -248.00000f);
	//glVertex3f(-84.000000f, 1140.6975f, -251.99998f);
	//glVertex3f(-84.000000f, 1140.0000f, -250.83081f);
	//glVertex3f(-87.999969f, 1144.0000f, -249.16913f);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glBegin(GL_LINES);
	glVertex3f(-86.0f, 1142.0f, -250.0f);
	glVertex3f(-86.0f + 0.73111796f, 1142.0f + 0.58590198f, -250.0f + 0.34955031f);
	glEnd();

#	define BUFFER_OFFSET(i) ((char *)NULL + (i))

	/*
	glColor3f(1.0f, 0.0f, 0.0f);
	for(int i=0;i<(int)m_bbList.size();i++)
	{
		drawBB(AABB(m_bbList[i].min, m_bbList[i].max));
	}
	*/

	/*
	Ray ray;
	ray.set(Vector3(-200.0f, 1000.0f, 50.0f), Vector3(0.759705f, -0.605159f, -0.237974f));
	HitPointInfo hit;
	Material mat;
	int hitIndex;
	AABB hitBB;
	unsigned int seed = tea<16>(0, 0);
	RayOctreeIntersect(ray, &AABB(m_octree.getHeader().min, m_octree.getHeader().max), 0, hit, hitBB, mat, hitIndex, 0.0f, 0.0f, seed, -1, 0);
	//RayOctreeIntersect(ray, &m_bbList[12], 0, hit, hitBB, mat, hitIndex, 0.0f, 0.0f, seed, 12, 0);

	//draw ray
	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(ray.origin().x(), ray.origin().y(), ray.origin().z());
	glVertex3f(ray.origin().x() + ray.direction().x() * 10000, ray.origin().y() + ray.direction().y() * 10000, ray.origin().z() + ray.direction().z() * 10000);
	glEnd();
	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_POINTS);
	glVertex3f(ray.origin().x(), ray.origin().y(), ray.origin().z());
	glEnd();
	*/
	/*
	for(int i=0;i<(int)m_bbList.size();i++)
	{
		render(i, AABB(m_octreeList[i].getHeader().min,m_octreeList[i].getHeader().max), 0, 1);
	}
	*/
	
	//render(camera, -1, AABB(m_octree.getHeader().min,m_octree.getHeader().max), 0, false, 1);
	//visVoxels(camera, -1, AABB(m_octree.getHeader().min,m_octree.getHeader().max), 0, false, 1);
	
	/*
	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex3f(-100.0f, 0.0f, -100.0f);
	glVertex3f(100.0f, 0.0f, -100.0f);
	glVertex3f(100.0f, 0.0f, 100.0f);
	glVertex3f(-100.0f, 0.0f, 100.0f);
	glEnd();
	glColor3f(1.0f, 1.0f, 0.0f);
	for(int i=0;i<10000;i++)
	{
		unsigned int seed = tea<16>(i, 0);
		Vector3 dir = Material::sampleAmbientOcclusionDirection(Vector3(0.0f, 1.0f, 0.0f), seed) * 100.0f;
		glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(dir.x(), dir.y(), dir.z());
		glEnd();
	}
	*/

#	if 0
	static FILE *fp = NULL;
	static Vector3 palette[256];
	if(!fp)
	{
		fp = fopen("palette.txt", "r");
		for(int i=0;i<256;i++)
		{
			fscanf(fp, "%f", &palette[i].e[0]);
			fscanf(fp, "%f", &palette[i].e[1]);
			fscanf(fp, "%f", &palette[i].e[2]);
		}
	}
	float scale = 100.0f;
	glLineWidth(2.0f);
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f*scale, 0.0f*scale, 0.0f*scale);
	glVertex3f(1.0f*scale, 0.0f*scale, 0.0f*scale);
	glEnd();
	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f*scale, 0.0f*scale, 0.0f*scale);
	glVertex3f(0.0f*scale, 1.0f*scale, 0.0f*scale);
	glEnd();
	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f*scale, 0.0f*scale, 0.0f*scale);
	glVertex3f(0.0f*scale, 0.0f*scale, 1.0f*scale);
	glEnd();
	glLineWidth(1.0f);
	glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
	drawBB(AABB(Vector3(0.0f*scale, 0.0f*scale, 0.0f*scale), Vector3(1.0f*scale, 1.0f*scale, 1.0f*scale)));
	glPointSize(2.0f);
	glBegin(GL_POINTS);
	for(int i=0;i<256;i++)
	{
		glColor3f(palette[i].e[0], palette[i].e[1], palette[i].e[2]);
		glVertex3f(palette[i].e[0]*scale, palette[i].e[1]*scale, palette[i].e[2]*scale);
	}
	glEnd();
	extern int g_numForDebug;
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	for(int i=0;i<g_numForDebug;i++)
	{
		glVertex3f(palette[i].e[0]*scale, palette[i].e[1]*scale, palette[i].e[2]*scale);
	}
	glEnd();
#	endif

#	ifdef VIS_PHOTONS
	glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
	glBegin(GL_POINTS);
	for(int i=0;i<m_numPhotons;i++)
	{
		glVertex3f(m_photons[i].pos.x(), m_photons[i].pos.y(), m_photons[i].pos.z());
	}
	glEnd();
#	endif

	render(&m_scene->getSceneGraph());
	//renderCulling();

	/*
	render(&m_scene->getSceneGraph());
	for(int i=0;i<m_scene->getNumEmitters();i++)
	{
		const Emitter &emitter = m_scene->getEmitter(i);
		if(emitter.type == Emitter::PARALLELOGRAM_LIGHT)
		{
			Vector3 v0, v1, v2, v3;
			v0 = emitter.planar.corner;
			v1 = v0 + emitter.planar.v1;
			v2 = v0 + emitter.planar.v1 + emitter.planar.v2;
			v3 = v0 + emitter.planar.v2;
			
			glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
			glBegin(GL_LINE_LOOP);
			glVertex3f(v0.x(), v0.y(), v0.z());
			glVertex3f(v1.x(), v1.y(), v1.z());
			glVertex3f(v2.x(), v2.y(), v2.z());
			glVertex3f(v3.x(), v3.y(), v3.z());
			glEnd();

			if((emitter.spotTarget.v1 + emitter.spotTarget.v2).maxAbsComponent() > 0.0f)
			{
				Vector3 v0, v1, v2, v3;
				v0 = emitter.spotTarget.corner;
				v1 = v0 + emitter.spotTarget.v1;
				v2 = v0 + emitter.spotTarget.v1 + emitter.spotTarget.v2;
				v3 = v0 + emitter.spotTarget.v2;

				glBegin(GL_LINE_LOOP);
				glVertex3f(v0.x(), v0.y(), v0.z());
				glVertex3f(v1.x(), v1.y(), v1.z());
				glVertex3f(v2.x(), v2.y(), v2.z());
				glVertex3f(v3.x(), v3.y(), v3.z());
				glEnd();
			}
		}
	}

	glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
	glBegin(GL_POINTS);
	for(int i=0;i<m_numPhotons;i++)
	{
		glVertex3f(m_photons[i].pos.x(), m_photons[i].pos.y(), m_photons[i].pos.z());
	}
	glEnd();
	*/

	/*
	//draw ray
	Ray ray;
	ray.set(Vector3(195.533493f, 3.673710f, -0.402110f), Vector3(-0.294545f, 0.950919f, 0.094852f));
	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(ray.origin().x(), ray.origin().y(), ray.origin().z());
	glVertex3f(ray.origin().x() + ray.direction().x() * 10000, ray.origin().y() + ray.direction().y() * 10000, ray.origin().z() + ray.direction().z() * 10000);
	glEnd();
	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_POINTS);
	glVertex3f(ray.origin().x(), ray.origin().y(), ray.origin().z());
	glEnd();
	*/

	//renderPhotonBB(0, m_bbPhotons, 0);

	if(image)
	{
		if(image->bpp == 4)
			glReadPixels(0, 0, image->width, image->height, GL_RGBA, GL_UNSIGNED_BYTE, image->data);
		else
			glReadPixels(0, 0, image->width, image->height, GL_RGB, GL_UNSIGNED_BYTE, image->data);
	}
}

void GLDebugging::render(SceneNode *sceneNode)
{
	if(sceneNode->model)
	{
		glPushMatrix();

		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		drawBB(sceneNode->nodeBB);

#		if 0
		Matrix mat = sceneNode->matrix;
		mat.transpose();
		glMultMatrixf((GLfloat*)mat.x);
		Model *model = sceneNode->model;
		int modelIndex = m_scene->getModelListMap()[model];
		GLfloat matAmbient[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		GLfloat matDiffuse[4] = {1.0f, 1.0f, 1.0f, 1.0f};
#		define SET_MATERIAL(mat, m) memcpy(mat, m, sizeof(float)*3)
		for(int j=0;j<model->getNumTriangles();j++)
		{
			Triangle *tri = model->getTriangle(j);
			SET_MATERIAL(matAmbient, model->getMaterial(tri->material).getMatKa().e);
			SET_MATERIAL(matDiffuse, model->getMaterial(tri->material).getMatKd().e);

			glColor4f(matAmbient[0] + matDiffuse[0], matAmbient[1] + matDiffuse[1], matAmbient[2] + matDiffuse[2], (matAmbient[3] + matDiffuse[3])/2);

			glBegin(GL_LINE_LOOP);
			glVertex3f(model->getVertex(tri->p[0])->v.x(), model->getVertex(tri->p[0])->v.y(), model->getVertex(tri->p[0])->v.z());
			glVertex3f(model->getVertex(tri->p[1])->v.x(), model->getVertex(tri->p[1])->v.y(), model->getVertex(tri->p[1])->v.z());
			glVertex3f(model->getVertex(tri->p[2])->v.x(), model->getVertex(tri->p[2])->v.y(), model->getVertex(tri->p[2])->v.z());
			glEnd();
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

void GLDebugging::render(Camera *camera, int voxel, const AABB &bb, int index, bool drawGeomBit, int depth)
{
	PhotonOctree *octree = voxel >= 0 ? &m_octreeList[voxel] : &m_octree;
	int N = m_octree.getHeader().dim;
	int childIndex = index * N * N * N;

	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				extern int g_numForDebug;
				AABB subBox = computeSubBox(x, y, z, bb);
				if(!(*octree)[childIndex].hasChild())
				{
					if(drawGeomBit && (*octree)[childIndex].isLeaf() || (g_numForDebug == depth && !(*octree)[childIndex].isEmpty()))
					{
						RGBf col = (*octree)[childIndex].getColorRef();
						//glColor4f(0.0f, 0.5f, 0.0f, 0.5f);
						//if(col.e[0] + col.e[1] + col.e[2] > 0.0f) 
						//{
						//	glColor4f(col.e[0], col.e[1], col.e[2], 1.0f);
						//}

						AABB childBox1;
						AABB childBox2;
						// level 1
						for(int x1=0,offset1=0;x1<N;x1++)
						{
							for(int y1=0;y1<N;y1++)
							{
								for(int z1=0;z1<N;z1++,offset1++)
								{
									const Voxel &voxel = (*octree)[childIndex];

									childBox1 = computeSubBox(x1, y1, z1, subBox);
									if(voxel.geomBitmap[offset1])
									{
										// level 2
										for(int x2=0,offset2=0;x2<N;x2++)
										{
											for(int y2=0;y2<N;y2++)
											{
												for(int z2=0;z2<N;z2++,offset2++)
												{
													childBox2 = computeSubBox(x2, y2, z2, childBox1);

													if(voxel.geomBitmap[offset1] & (1u << offset2))
													{
														Vector3 mid = 0.5f*(childBox2.min + childBox2.max);
														float val = (camera->getEye() - mid).length() / 200.0f;
														//printf("val = %f\n", val);
														glColor3f(min(val, 1.0f), min(val, 1.0f), 0.0f);
														drawBB(childBox2, true);
														//glColor3f(rand()/(float)RAND_MAX, rand()/(float)RAND_MAX, rand()/(float)RAND_MAX);
														glBegin(GL_POINTS);
														glVertex3f(mid.x(), mid.y(), mid.z());
														glEnd();
													}
												}
											}
										}
									}
								}
							}
						}


						/*
						GLfloat mat[3] = {0.0f, 0.5f, 0.0f};
						glMaterialfv(GL_FRONT, GL_DIFFUSE, mat);
						glBegin(GL_TRIANGLES);
						glNormal3f(octree[childIndex].getNorm().e[0], octree[childIndex].getNorm().e[1], octree[childIndex].getNorm().e[2]);
						glVertex3f(octree[childIndex].corners[0].e[0], octree[childIndex].corners[0].e[1], octree[childIndex].corners[0].e[2]);
						glVertex3f(octree[childIndex].corners[1].e[0], octree[childIndex].corners[1].e[1], octree[childIndex].corners[1].e[2]);
						glVertex3f(octree[childIndex].corners[2].e[0], octree[childIndex].corners[2].e[1], octree[childIndex].corners[2].e[2]);
						glEnd();
						glBegin(GL_TRIANGLES);
						glNormal3f(octree[childIndex].getNorm().e[0], octree[childIndex].getNorm().e[1], octree[childIndex].getNorm().e[2]);
						glVertex3f(octree[childIndex].corners[2].e[0], octree[childIndex].corners[2].e[1], octree[childIndex].corners[2].e[2]);
						glVertex3f(octree[childIndex].corners[1].e[0], octree[childIndex].corners[1].e[1], octree[childIndex].corners[1].e[2]);
						glVertex3f(octree[childIndex].corners[3].e[0], octree[childIndex].corners[3].e[1], octree[childIndex].corners[3].e[2]);
						glEnd();
						*/
						
						//drawBB(subBox, true);
					}
				}
				else
				{
					switch(depth%3)
					{
					case 1: glColor4f(1.0f, 0.0f, 0.0f, 0.3f); break;
					case 2: glColor4f(0.0f, 1.0f, 0.0f, 0.3f); break;
					case 0: glColor4f(0.0f, 0.0f, 1.0f, 0.3f); break;
					//default: glColor4f(0.0f, 1.0f, 0.0f, 0.3f); break;
					}

					//glColor3f(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
					//if(g_numForDebug == depth) 
					drawBB(subBox, false);

					bool d = drawGeomBit;
					if(childIndex == 0) d = true;
					if((*octree)[childIndex].hasChild() && g_numForDebug >= depth) 
						render(camera, voxel, subBox, (*octree)[childIndex].getChildIndex(), d, depth+1);
				}
				childIndex++;
			}
}

void GLDebugging::visVoxels(Camera *camera, int voxel, const AABB &bb, int index, bool drawGeomBit, int depth)
{
	glEnable(GL_DEPTH_TEST);					

	PhotonOctree *octree = voxel >= 0 ? &m_octreeList[voxel] : &m_octree;
	int N = m_octree.getHeader().dim;
	int childIndex = index * N * N * N;

	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				extern int g_numForDebug;
				AABB subBox = computeSubBox(x, y, z, bb);

				if(g_numForDebug == depth)
				{
					//glColor3f(0.8f, 0.8f, 0.8f);
					drawBB(subBox, true);
					glColor3f(0.0f, 0.0f, 0.0f);
					//glColor3f(1.0f, 1.0f, 1.0f);
					drawBB(subBox, false);
				}

				if(!(*octree)[childIndex].hasChild())
				{
				}
				else
				{
					bool d = drawGeomBit;
					if(childIndex == 0) d = true;
					if((*octree)[childIndex].hasChild() && g_numForDebug > depth) 
						visVoxels(camera, voxel, subBox, (*octree)[childIndex].getChildIndex(), d, depth+1);
				}
				childIndex++;
			}
}

void GLDebugging::renderCut(const AABB &bb, int index, const Vector3 &from)
{
	int N = m_octree.getHeader().dim;
	int childIndex = index * N * N * N;

	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				AABB subBox = computeSubBox(x, y, z, bb);


				if(!m_octree[childIndex].hasChild())
				{
					if(m_octree[childIndex].isLeaf())
					{
						drawBB(subBox, true);
					}
				}
				else
				{
					renderCut(subBox, m_octree[childIndex].getChildIndex(), from);
				}
				childIndex++;
			}
}

AABB GLDebugging::computeSubBox(int x, int y, int z, const AABB &box)
{
	AABB subBox;
	/*
	float boxDelta = (box.max.x() - box.min.x()) / m_octree.getHeader().dim;
	subBox.min = box.min + (Vector3((float)x, (float)y, (float)z)*boxDelta);
	subBox.max = subBox.min + Vector3(boxDelta);
	*/
	subBox = box;
	*(x ? &subBox.min.e[0] : &subBox.max.e[0]) = 0.5f*(subBox.min.e[0] + subBox.max.e[0]);
	*(y ? &subBox.min.e[1] : &subBox.max.e[1]) = 0.5f*(subBox.min.e[1] + subBox.max.e[1]);
	*(z ? &subBox.min.e[2] : &subBox.max.e[2]) = 0.5f*(subBox.min.e[2] + subBox.max.e[2]);
	return subBox;
}

void GLDebugging::tracePhotons()
{
	int numMaxPhotons = m_scene->getNumMaxPhotons();
	if(m_photons)
		delete[] m_photons;
	m_photons = new Photon[numMaxPhotons];
	printf("num tracing photons = %d\n", numMaxPhotons);
	m_numPhotons = m_scene->tracePhotons(numMaxPhotons, m_photons, m_octree.addPhoton);
	printf("num valid photons = %d\n", m_numPhotons);
	/*
	FILE *fp = fopen("photons", "r");
	for(int i=0;i<numMaxPhotons;i++)
	{
		fscanf(fp, "%d", &m_photons[i].axis);
		fscanf(fp, "%f", &m_photons[i].pos.e[0]);
		fscanf(fp, "%f", &m_photons[i].pos.e[1]);
		fscanf(fp, "%f", &m_photons[i].pos.e[2]);
	}
	m_numPhotons = numMaxPhotons;
	fclose(fp);
	*/
}

void GLDebugging::buildPhotonKDTree()
{
	m_scene->buildPhotonKDTree(m_numPhotons, &m_photons, m_bbPhotons);
}

void GLDebugging::renderPhotonBB(int curNode, const AABB &curBB, int depth)
{
	if(curNode >= m_numPhotons) 
		return ;

	int axis = m_photons[curNode].axis;

	if(axis == Photon::SPLIT_AXIS_LEAF || axis == Photon::SPLIT_AXIS_NULL) return;

	switch(depth%3)
	{
	case 0 : glColor4f(1.0f, 0.0f, 0.0f, 0.3f); break;
	case 1 : glColor4f(0.0f, 1.0f, 0.0f, 0.3f); break;
	case 2 : glColor4f(0.0f, 0.0f, 1.0f, 0.3f); break;
	}

	drawBB(curBB);

	AABB leftBB(curBB), rightBB(curBB);
	leftBB.max.e[axis] = m_photons[curNode].pos.e[axis];
	rightBB.min.e[axis] = m_photons[curNode].pos.e[axis];

	renderPhotonBB(curNode*2+1, leftBB, depth+1);
	renderPhotonBB(curNode*2+2, rightBB, depth+1);
}

int GLDebugging::frustumBoxIntersect(int numPlane, const Plane plane[], const AABB &bb)
{
	int ret = 1;	// 0 : intersect, 1 : inside, 2 : outside, 3 : far
	Vector3 vmin, vmax; 

	for(int i=0;i<numPlane;i++) { 
		// X axis
		if(plane[i].n.e[0] > 0) {
			vmin.e[0] = bb.min.e[0];
			vmax.e[0] = bb.max.e[0];
		} else {
			vmin.e[0] = bb.max.e[0];
			vmax.e[0] = bb.min.e[0];
		}
		// Y axis
		if(plane[i].n.e[1] > 0) {
			vmin.e[1] = bb.min.e[1];
			vmax.e[1] = bb.max.e[1];
		} else {
			vmin.e[1] = bb.max.e[1];
			vmax.e[1] = bb.min.e[1];
		}
		// Z axis
		if(plane[i].n.e[2] > 0) {
			vmin.e[2] = bb.min.e[2];
			vmax.e[2] = bb.max.e[2];
		} else {
			vmin.e[2] = bb.max.e[2];
			vmax.e[2] = bb.min.e[2];
		}
		if(dot(plane[i].n, vmin) - plane[i].d > 0)
			return i == 5 ? 3 : 2;
		if(dot(plane[i].n, vmax) - plane[i].d >= 0)
			ret = 0;
	}
	return ret;
}

int g_numInside;
int g_numOutside;
int g_numIntersect;
int g_numFar;

void GLDebugging::renderCulling()
{
	Camera *camera = OpenIRT::getSingletonPtr()->getCameraMap()["Camera5"];
	Ray frustumRay[4];
	camera->getRayWithOrigin(frustumRay[0], 0.0f, 0.0f);
	camera->getRayWithOrigin(frustumRay[1], 1.0f, 0.0f);
	camera->getRayWithOrigin(frustumRay[2], 1.0f, 1.0f);
	camera->getRayWithOrigin(frustumRay[3], 0.0f, 1.0f);

	glColor3f(1.0f, 1.0f, 0.0f);
	for(int i=0;i<4;i++)
	{
		glBegin(GL_LINES);
		glVertex3f(frustumRay[i].origin().x(), frustumRay[i].origin().y(), frustumRay[i].origin().z());
		glVertex3f(frustumRay[i].origin().x() + frustumRay[i].direction().x() * 10000, frustumRay[i].origin().y() + frustumRay[i].direction().y() * 10000, frustumRay[i].origin().z() + frustumRay[i].direction().z() * 10000);
		glEnd();
	}

}

int getNumTris(TEST_MODEL_TYPE *model, int index)
{
	BVHNode *node = model->getBV(index);
	if(model->isLeaf(node))
	{
		return model->getNumTriangles(node);
	}

	return getNumTris(model, model->getLeftChildIdx(node)) + getNumTris(model, model->getRightChildIdx(node));
}

void GLDebugging::renderCulling(const Plane plane[], int index, int depth)
{
	TEST_MODEL_TYPE *model = (TEST_MODEL_TYPE *)m_scene->getModelList()[0];
	BVHNode *node = model->getBV(index);

	extern int g_numForDebug;

	AABB bb(node->min, node->max);

	int intersect = frustumBoxIntersect(6, plane, bb);
	switch(intersect)
	{
	case 0: glColor3f(1.0f, 0.0f, 0.0f); break;
	case 1: glColor3f(0.0f, 1.0f, 0.0f); break;
	case 2: glColor3f(0.0f, 0.0f, 1.0f); break;
	case 3: glColor3f(0.0f, 1.0f, 1.0f); break;
	}

	//if(depth == g_numForDebug)
	{
		drawBB(bb);
		switch(intersect)
		{
		case 0 : g_numIntersect ++; break;
		case 1 : g_numInside ++; break;
		case 2 : g_numOutside ++; break;
		case 3 : g_numFar ++; break;
		}
		/*
		switch(intersect)
		{
		case 0 : g_numIntersect += getNumTris(model, index); break;
		case 1 : g_numInside += getNumTris(model, index); break;
		case 2 : g_numOutside += getNumTris(model, index); break;
		case 3 : g_numFar += getNumTris(model, index); break;
		}
		*/
	}

	if(!model->isLeaf(node) && depth < g_numForDebug)
	{
		renderCulling(plane, model->getLeftChildIdx(node), depth+1);
		renderCulling(plane, model->getRightChildIdx(node), depth+1);
	}
}

#define BSP_EPSILON 0.001f
#define INTERSECT_EPSILON 0.01f
#define TRI_INTERSECT_EPSILON 0.0001f
bool GLDebugging::RayLODIntersect(const Ray &ray, const AABB &bb, const Voxel &voxel, HitPointInfo &hit, Material &material, float tmax, unsigned int seed)
{
	Vector3 norm;
	//norm = voxel.norm;
	norm = voxel.getNorm();
	float vdot = norm.x()*ray.direction().x() + norm.y()*ray.direction().y() + norm.z()*ray.direction().z();
	float vdot2 = norm.x()*ray.origin().x() + norm.y()*ray.origin().y() + norm.z()*ray.origin().z();
	float t = (voxel.d - vdot2) / vdot;

	// if either too near or further away than a previous hit, we stop
	if (t < (0-INTERSECT_EPSILON*10) || t > (tmax + INTERSECT_EPSILON*10))
		return false;	

	if (t < (bb.max.x() - bb.min.x())*3 && t < (bb.max.y() - bb.min.y())*3 && t < (bb.max.z() - bb.min.z())*3)
		return false;

	/*
	material.mat_Kd = voxel.getKd();
	material.mat_Ks = voxel.getKs();
	material.mat_d = voxel.getD();
	material.mat_Ns = voxel.getNs();
	material.recalculateRanges();
	*/
	if(material.isRefraction(voxel.getD(), seed)) return false;

	// we have a hit:
	// fill hitpoint structure:
	//
	hit.m = 0;
	hit.t = t;
	hit.n = vdot > 0 ? -norm : norm;
	return true;	
}

bool GLDebugging::RayOctreeIntersect(const Ray &ray, AABB *startBB, int startIdx, HitPointInfo &hit, AABB &hitBB, Material &material, int &hitIndex, float limit, float tLimit, unsigned int seed, int x, int y)
{
	const OctreeHeader &c_octreeHeader = m_octree.getHeader();
	AABB bb;
	if(startBB)
	{
		bb.min = startBB->min;
		bb.max = startBB->max;
	}
	else
	{
		bb.min = c_octreeHeader.min;
		bb.max = c_octreeHeader.max;
	}

	float t0, t1;
	if(!ray.boxIntersect(bb.min, bb.max, t0, t1)) return false;

	int flag = 0;

	Vector3 ori = ray.origin(), dir = ray.direction();
	Vector3 temp = bb.max + bb.min;

	if(ray.direction().x() < 0.0f)
	{
		ori.e[0] = temp.e[0] - ori.e[0];
		dir.e[0] = -dir.e[0];
		flag |= 4;
	}

	if(ray.direction().y() < 0.0f)
	{
		ori.e[1] = temp.e[1] - ori.e[1];
		dir.e[1] = -dir.e[1];
		flag |= 2;
	}

	if(ray.direction().z() < 0.0f)
	{
		ori.e[2] = temp.e[2] - ori.e[2];
		dir.e[2] = -dir.e[2];
		flag |= 1;
	}

	Ray transRay;
	transRay.set(ori, dir);

	AABB newBB;
	newBB.min = (bb.min - transRay.origin()) * transRay.invDirection();
	newBB.max = (bb.max - transRay.origin()) * transRay.invDirection();

	int N = c_octreeHeader.dim;

	typedef struct tempStack_t
	{
		int childIndex;
		int child;
		AABB bb;
	} tempStack;

	Vector3 mid;
	Voxel root;
	root.setChildIndex(startIdx);
	tempStack stack[100];

	int stackPtr;

	/*
#	define FIRST_NODE(bb, mid, child) child = 0;\
	if ((bb)->min.y < (bb)->min.x && (bb)->min.z < (bb)->min.x) {if ((mid).y < (bb)->min.x) child |= 2;if ((mid).z < (bb)->min.x) child |= 1;} \
	else if ((bb)->min.z < (bb)->min.y) {if ((mid).x < (bb)->min.y) child |= 4;if ((mid).z < (bb)->min.y) child |= 1;} \
	else {if ((mid).x < (bb)->min.z) child |= 4;if ((mid).y < (bb)->min.z) child |= 2;}
	*/
	mid = 0.5f*(newBB.min + newBB.max);

	/*
	stack[0].bb = newBB;
	stack[0].index = -1;
	//FIRST_NODE(&newBB, mid, stack[0].child);
	stackPtr = 1;
	*/
	stackPtr = 0;

	AABB currentBB = newBB;
	Voxel currentVoxel = root;

	int child = -1, childIndex = startIdx;

	while(true)
	{
		if(x == 123123) printf("childIndex = %d\n", childIndex);

		static int ss = 0;
		switch((ss++) % 3)
		{
		case 0: glColor3f(1.0f, 0.0f, 0.0f); break;
		case 1: glColor3f(0.0f, 1.0f, 0.0f); break;
		case 2: glColor3f(0.0f, 0.0f, 1.0f); break;
		}

		AABB realBB;
		realBB.min = (currentBB.min * transRay.direction()) + transRay.origin();
		realBB.max = (currentBB.max * transRay.direction()) + transRay.origin();
		drawBB(realBB);

		//if(!currentVoxel.isEmpty() && !(currentBB.max.x < 0.0f || currentBB.max.y < 0.0f || currentBB.max.z < 0.0f))
		if(!currentVoxel.isEmpty() && 
			currentBB.max.x() > 0.0f && currentBB.max.y() > 0.0f && currentBB.max.z() > 0.0f &&
			currentBB.min.x() < hit.t && currentBB.min.y() < hit.t && currentBB.min.y() < hit.t)
		{
			if(currentVoxel.isLeaf())
			{
				//if(RayLODIntersect(ray, currentVoxel, hit, material, fminf(currentBB.max.minComponent(), hit.t), seed))
				if(RayLODIntersect(ray, currentBB, currentVoxel, hit, material, hit.t, seed))
				{
					hitIndex = childIndex + (child^flag);
					hitBB = currentBB;
					if(x == 123123) printf("hitIndex = %d\n", hitIndex);
					return true;
				}
				goto NEXT_SIBILING;
			}

			// push down

			//FIRST_NODE(currentBB, mid, child);
			// get first child
			child = 0;
			if (currentBB.min.y() < currentBB.min.x() && currentBB.min.z() < currentBB.min.x())
			{
				// YZ plane
				if (mid.y() < currentBB.min.x()) child |= 2;
				if (mid.z() < currentBB.min.x()) child |= 1;
			}
			else if (currentBB.min.z() < currentBB.min.y())
			{
				// XZ plane
				if (mid.x() < currentBB.min.y()) child |= 4;
				if (mid.z() < currentBB.min.y()) child |= 1;
			}
			else
			{
				// XY plane
				if (mid.x() < currentBB.min.z()) child |= 4;
				if (mid.y() < currentBB.min.z()) child |= 2;
			}

			childIndex = currentVoxel.getChildIndex() * N * N * N;

			stack[stackPtr].bb = currentBB;
			stack[stackPtr].childIndex = childIndex;
			stack[stackPtr++].child = child;

			currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
			currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
			currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
			currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
			currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
			currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
			mid = 0.5f*(currentBB.min + currentBB.max);

			/*
			VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
			VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
			*/
			if(x >= 0)
				currentVoxel = m_octreeList[x][childIndex + (child^flag)];
			else
				currentVoxel = m_octree[childIndex + (child^flag)];

			continue;
		}

		// move to next sibiling
NEXT_SIBILING : 

		if(stackPtr < 1) return false;
		// get stack top
		childIndex = stack[stackPtr-1].childIndex;
		currentBB = stack[stackPtr-1].bb;
		child = stack[stackPtr-1].child;
		mid = 0.5f*(currentBB.min + currentBB.max);

#		define NEW_NODE(x, y, z, a, b, c) (((x) < (y) && (x) < (z)) ? (a) : (((y) < (z)) ? (b) : (c)))

		switch(child)
		{
		case 0 : child = NEW_NODE(mid.x(), mid.y(), mid.z(), 4, 2, 1); break;
		case 1 : child = NEW_NODE(mid.x(), mid.y(), currentBB.max.z(), 5, 3, 8); break;
		case 2 : child = NEW_NODE(mid.x(), currentBB.max.y(), mid.z(), 6, 8, 3); break;
		case 3 : child = NEW_NODE(mid.x(), currentBB.max.y(), currentBB.max.z(), 7, 8, 8); break;
		case 4 : child = NEW_NODE(currentBB.max.x(), mid.y(), mid.z(), 8, 6, 5); break;
		case 5 : child = NEW_NODE(currentBB.max.x(), mid.y(), currentBB.max.z(), 8, 7, 8); break;
		case 6 : child = NEW_NODE(currentBB.max.x(), currentBB.max.y(), mid.z(), 8, 8, 7); break;
		case 7 : child = 8;	break;
		}
		stack[stackPtr-1].child = child;

		if(child == 8)
		{
			if (--stackPtr == 0) break;
			goto NEXT_SIBILING;
		}

		/*
		VOXEL_FETCH(childIndex + (child^flag), 0, currentVoxel.low);
		VOXEL_FETCH(childIndex + (child^flag), 1, currentVoxel.high);
		*/
		if(x >= 0)
			currentVoxel = m_octreeList[x][childIndex + (child^flag)];
		else
			currentVoxel = m_octree[childIndex + (child^flag)];
		currentBB.min.e[0] = (child & 0x4) ? mid.e[0] : currentBB.min.e[0];
		currentBB.min.e[1] = (child & 0x2) ? mid.e[1] : currentBB.min.e[1];
		currentBB.min.e[2] = (child & 0x1) ? mid.e[2] : currentBB.min.e[2];
		currentBB.max.e[0] = (child & 0x4) ? currentBB.max.e[0] : mid.e[0];
		currentBB.max.e[1] = (child & 0x2) ? currentBB.max.e[1] : mid.e[1];
		currentBB.max.e[2] = (child & 0x1) ? currentBB.max.e[2] : mid.e[2];
		mid = 0.5f*(currentBB.min + currentBB.max);
	}
	return false;
}