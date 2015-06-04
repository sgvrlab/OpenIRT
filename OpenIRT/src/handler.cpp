/********************************************************************
	created:	2010/02/05
	file path:	D:\Projects\Redering\OpenIRT\src
	file base:	handler
	file ext:	cpp
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Handler has global variables for handling parameters
*********************************************************************/

#include "defines.h"
#include "CommonHeaders.h"
#include "OpenIRT.h"

// Handlers for glut
int g_windowHandle;
int g_handleMenu, 
g_handleRenderingModeMenu,
g_handleDebugMenu,
g_handleLoadSceneMenu;

// Screen width and height
bool g_resized = false;
int g_frame = 1;	// complete image frame
int g_frame2 = 1;	// number of reset of g_frame
int g_frame3 = 1;	// number of frame (a block of complete image frame)
int g_frame4 = 1;	// number of complete image frames from fixed view point
float g_frameTime = 0.0f;
float g_frameSumTime = 0.0f;
int g_frameNumSum = 0;
float g_frame3Time = 0.0f;
bool g_rendering = false;
int g_mouseX;
int g_mouseY;
bool g_cameraMoving = false;
bool g_isRendering = false;
float g_responseTime = 0.0f;
int g_statTryCount = 2048;
//
// Variables for scenes
//

//char g_currentBaseDir[MAX_PATH];
char g_currentSceneFileName[MAX_PATH];
char g_currentPathFileName[MAX_PATH];
char g_currentCameraFileName[MAX_PATH];
char g_currentScreenShotPathName[MAX_PATH];
char g_workingDirectory[MAX_PATH];
FileList g_sceneFileList;

RenderingMode g_renderMode = RENDER_MODE_PATH_TRACING;

bool g_usePhotonMapping = false;
bool g_preparedRender = false;
bool g_useCustomBitmap = true;

int g_numForDebug = 1;		// use only for debug

// timers
int g_timerFPS;
int g_timerPhotonTracing;
int g_timerBuildingPhotonKDTree;
int g_timerConverge;