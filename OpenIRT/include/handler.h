/********************************************************************
	created:	2010/02/03
	file path:	d:\Projects\Redering\OpenIRT\include
	file base:	handler
	file ext:	h
	author:		Tae-Joon Kim (tjkim@tclab.kaist.ac.kr)
	
	comment:	Handler has global variables for handling parameters
*********************************************************************/

#pragma once

// Handlers for glut
extern int g_windowHandle;
extern int g_handleMenu, 
g_handleRenderingModeMenu,
g_handleDebugMenu,
g_handleLoadSceneMenu;

// Option and logs manager for runtime options
// Screen width and height
extern bool g_resized;
extern int g_frame;
extern int g_frame2;
extern int g_frame3;
extern int g_frame4;
extern float g_frameTime;
extern float g_frameSumTime;
extern int g_frameNumSum;
extern float g_frame3Time;
extern bool g_rendering;
extern int g_mouseX;
extern int g_mouseY;
extern bool g_cameraMoving;
extern bool g_isRendering;
extern float g_responseTime;
extern int g_statTryCount;

//
// Variables for scenes
//
//extern char g_currentBaseDir[MAX_PATH];
extern char g_currentSceneFileName[MAX_PATH];
extern char g_currentPathFileName[MAX_PATH];
extern char g_currentCameraFileName[MAX_PATH];
extern char g_currentScreenShotPathName[MAX_PATH];
extern char g_workingDirectory[MAX_PATH];
extern FileList g_sceneFileList;

extern RenderingMode g_renderMode;
extern bool g_usePhotonMapping;

extern bool g_preparedRender;
extern bool g_useCustomBitmap;

// timers
extern int g_timerFPS;
extern int g_timerPhotonTracing;
extern int g_timerBuildingPhotonKDTree;
extern int g_timerConverge;