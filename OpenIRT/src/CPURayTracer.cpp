#include "CommonOptions.h"
#include "CPURayTracer.h"

using namespace irt;

CPURayTracer::CPURayTracer(void)
{
}

CPURayTracer::~CPURayTracer(void)
{
	done();
}

void CPURayTracer::init(Scene *scene)
{
	Renderer::init(scene);

	if(!scene) return;

	done();

	m_intersectionStream = scene->getIntersectionStream();

	sceneChanged();
}

void CPURayTracer::done()
{
}

void CPURayTracer::resized(int width, int height)
{
}

void CPURayTracer::sceneChanged()
{
}

void CPURayTracer::materialChanged()
{
}

#include "HCCMesh.h"
#include "HCCMesh2.h"
void CPURayTracer::render(Camera *camera, Image *image, unsigned int seed)
{
#	if 1
	static int timer = StopWatch::create();
	
	//
	// set up tiling:
	//
	int tileWidth = 16;
	int tileHeight = 16;
	int tilesX = image->width / tileWidth;
	int tilesY = image->height / tileHeight;
	int numTiles = tilesX * tilesY;	

	StopWatch::get(timer).reset();
	StopWatch::get(timer).start();
#	pragma omp parallel for schedule(dynamic)
	for (int curTile = 0; curTile < numTiles; curTile++)
	{		
		unsigned int startX = (curTile % tilesX) * tileWidth;
		unsigned int startY = (unsigned int)(curTile / tilesY) * tileHeight;		

		// normale single ray tracing:
		float deltaX = 1.0f / (float)image->width;
		float deltaY = 1.0f / (float)image->height;

		Ray ray;
		RGB4f outColor;

		float ypos = deltaY/2.0f + startY*deltaY,
			xpos = deltaX/2.0f + startX*deltaX;

		for (int y = 0; y < tileHeight; y++) 
		{				
			xpos = deltaX/2.0f + startX*deltaX;
			for (int x = 0; x < tileWidth; x++) {									

				camera->getRayWithOrigin(ray, xpos, ypos);

				//to visualize normal, position, and principal directions.

				m_scene->trace(ray, outColor);

				image->setPixel(startX + x, (image->height - (startY + y) - 1), RGBf(outColor.e));

				xpos += deltaX;
			}
			ypos += deltaY;
		}
	}
	StopWatch::get(timer).stop();
#	else
	static int timer = StopWatch::create();
	
	static const int nRaysPerSide = TILE_SIZE/2;
	static const int nRealRaysPerSide = TILE_SIZE;
	static const int nRays = nRaysPerSide*nRaysPerSide;
	static const int nRealRays = nRealRaysPerSide*nRealRaysPerSide;
	int numSubPixelsX = image->width;
	int numSubPixelsY = image->height;
	int numPacketsX = numSubPixelsX / nRealRaysPerSide;
	int numPacketsY = numSubPixelsY / nRealRaysPerSide;
	int numPackets = numPacketsX * numPacketsY;

	Vector3 corner2 = camera->getCorner2();

	float deltax = 1.0f / (float)numSubPixelsX;
	float deltay = 1.0f / (float)numSubPixelsY;

	//static int numThreads = omp_get_max_threads();

////#	define USE_SINGLE_THREAD
//#	ifdef USE_SINGLE_THREAD
//	omp_set_num_threads(1);
//#	else
//	omp_set_num_threads(max(1, numThreads-1));
//#	endif

	StopWatch::get(timer).reset();
	StopWatch::get(timer).start();
#	pragma omp parallel for schedule(dynamic)
	for (int curPacket = 0; curPacket < numPackets; curPacket++) 
	{		
		RayPacket<nRays, true, true, true> rayPacket;
		RayPacket<nRays, false, true, true> *rayPacketNonCoherent = (RayPacket<nRays, false, true, true> *)((void *)&rayPacket);

		__declspec(align(16)) RGB4f colors[nRealRays];
		//__declspec(align(16)) RGB4f colors2[nRealRays];

		for(int i=0;i<nRealRays;i++) colors[i].set(RGBf(), 1.0f);

		unsigned int start_x = (curPacket % numPacketsX)*nRealRaysPerSide;
		unsigned int start_y = (curPacket / numPacketsX)*nRealRaysPerSide;
		
		float ypos = start_y*deltay,
			xpos = start_x*deltax;

		__declspec(align(16)) float jitter[nRealRays][2] = {0, };

		// initialize beam with real eye-rays:
		rayPacket.setupForPrimaryRays(camera->getEye(), corner2, camera->getScaledRight(), camera->getScaledUp(), nRaysPerSide, nRaysPerSide, xpos, ypos, deltax, deltay, jitter);
		
		//if (rayPacket.hasMatchingDirections())
			//m_scene->trace(rayPacket, colors, 0, m_intersectionStream);
		//else
			m_scene->trace(*rayPacketNonCoherent, colors, 0, m_intersectionStream);

		// fill ray & hit information from packet
		for(int i=0;i<nRealRays;i++)
		{
			unsigned int sX = i % nRealRaysPerSide;
			unsigned int sY = i / nRealRaysPerSide;
			unsigned int offset = (sY/2)*nRaysPerSide+(sX/2);
			unsigned int offset2 = (sY%2)*2+(sX%2);

			unsigned int x = start_x + sX;
			unsigned int y = image->height - (start_y + sY) - 1;

			/*
			RGBf color = colors[offset*4 + offset2];
			float temp = color.e[2];
			color.e[2] = color.e[0];
			color.e[0] = temp;
			*/

			RGB4f color = colors[offset*4 + offset2];
			
			image->setPixel(x, y, color);
		}
	}
	StopWatch::get(timer).stop();
	//printf("numHits = %d\n", numHits);
#	endif
}
