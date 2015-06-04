#ifndef COMMON_OPTIONS_H
#define COMMON_OPTIONS_H

//#define USE_CPU_GLOSSY
//#define TEST_VOXEL
//#define VIS_PHOTONS
//#define DISABLE_GPU
//#define USE_FULL_DETAIL_GI
//#define TEST_GLOSSY
//#define USE_PLANE_D

#define USE_PARAM_OPTIMIZATION
#define USE_DUAL_BUFFER
#define USE_PREVIEW
//#define TRACE_PHOTONS
//#define USE_HCCMESH_QUANTIZATION
#define USE_OOCVOXEL
#define OOCVOXEL_SUPER_RESOLUTION 0.25f
//#define USE_SINGLE_THREAD
//#define USE_VOXEL_LOD
//#define USE_VERTEX_NORMALS
//#define USE_LIGHT_COSINE
//#define USE_GAUSSIAN_RECONSTRUCTION
#define USE_ATOMIC_OPERATIONS
#define USE_PACKET
//#define USE_TEXTURING
//#define USE_2ND_RAYS_FILTER
//#define USE_MM

#define USE_PHONG_HIGHLIGHTING
#define EXTRACT_IMAGE_DEPTH
#define EXTRACT_IMAGE_NORMAL

#define TILE_SIZE 8
//#define STAT_TRY_COUNT 128

#define PHOTON_INTENSITY_SCALING_FACTOR 1.0f
#define PHOTON_INTENSITY_SCALING_FACTOR_FULL_DETAIL 0.4f

#define INTERSECT_EPSILON 0.01f
#define TRI_INTERSECT_EPSILON 0.0001f

//#define PACKET_EPSILON 1.0f
#define PACKET_EPSILON 0.05f

#ifndef USE_OOCVOXEL
#undef OOCVOXEL_SUPER_RESOLUTION
#define OOCVOXEL_SUPER_RESOLUTION 1.0f
#endif

#define USE_PRIORITY_DISTANCE 1
#define USE_PRIORITY_REQUESTED_COUNT 2
#define VOXEL_PRIORITY_POLICY USE_PRIORITY_DISTANCE

#ifndef fminf
#define fminf(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef fmaxf
#define fmaxf(a,b) (((a) > (b)) ? (a) : (b))
#endif

#define MAX_NUM_OPENGL_RESOURCES 16
#define MAX_NUM_EMITTERS 16
#define MAX_NUM_MODELS 64

#ifdef USE_FULL_DETAIL_GI
#define tracePhotonsAPI tracePhotonsAcc
#define traceSubPhotonsToBackBufferAPI traceSubPhotonsToBackBufferAcc
#define swapPhotonBufferAPI swapPhotonBufferAcc
#define buildPhotonKDTreeAPI buildPhotonKDTreeAcc
#define lightChangedAPI lightChangedAcc
#define materialChangedAPI materialChangedAcc
#define loadSceneAPI loadSceneAcc
#define updateControllerAPI updateControllerAcc
#define renderBeginAPI renderBeginAcc
#define renderPartAPI renderPartAcc
#define renderEndAPI renderEndAcc
#define resetAPI resetAcc
#define initWithImageAPI initWithImageAcc
#define loadOOCVoxelInfoAPI loadOOCVoxelInfoAcc
#define OnVoxelChangedAPI OnVoxelChangedAcc
#define loadVoxelsAPI loadVoxelsAcc
#define loadPhotonVoxelsAPI loadPhotonVoxelsAcc
#define beginGatheringRequestCountAPI beginGatheringRequestCountAcc
#define endGatheringRequestCountAPI endGatheringRequestCountAcc
#define updateControllerAPI updateControllerAcc
#define unloadSceneAPI unloadSceneAcc
#elif defined(DISABLE_GPU)
#define tracePhotonsAPI tracePhotonsEmu
#define traceSubPhotonsToBackBufferAPI traceSubPhotonsToBackBufferEmu
#define swapPhotonBufferAPI swapPhotonBufferEmu
#define buildPhotonKDTreeAPI buildPhotonKDTreeEmu
#define lightChangedAPI lightChangedEmu
#define materialChangedAPI materialChangedEmu
#define loadSceneAPI loadSceneEmu
#define updateControllerAPI updateControllerEmu
#define renderBeginAPI renderBeginEmu
#define renderPartAPI renderPartEmu
#define renderEndAPI renderEndEmu
#define resetAPI resetEmu
#define initWithImageAPI initWithImageEmu
#define loadOOCVoxelInfoAPI loadOOCVoxelInfoEmu
#define OnVoxelChangedAPI OnVoxelChangedEmu
#define loadVoxelsAPI loadVoxelsEmu
#define loadPhotonVoxelsAPI loadPhotonVoxelsEmu
#define beginGatheringRequestCountAPI beginGatheringRequestCountEmu
#define endGatheringRequestCountAPI endGatheringRequestCountEmu
#define updateControllerAPI updateControllerEmu
#define unloadSceneAPI unloadSceneEmu
#else
#define tracePhotonsAPI tracePhotonsTReX
#define traceSubPhotonsToBackBufferAPI traceSubPhotonsToBackBufferTReX
#define swapPhotonBufferAPI swapPhotonBufferTReX
#define buildPhotonKDTreeAPI buildPhotonKDTreeTReX
#define lightChangedAPI lightChangedTReX
#define materialChangedAPI materialChangedTReX
#define loadSceneAPI loadSceneTReX
#define updateControllerAPI updateControllerTReX
#define renderBeginAPI renderBeginTReX
#define renderPartAPI renderPartTReX
#define renderEndAPI renderEndTReX
#define resetAPI resetTReX
#define initWithImageAPI initWithImageTReX
#define loadOOCVoxelInfoAPI loadOOCVoxelInfoTReX
#define OnVoxelChangedAPI OnVoxelChangedTReX
#define loadVoxelsAPI loadVoxelsTReX
#define loadPhotonVoxelsAPI loadPhotonVoxelsTReX
#define beginGatheringRequestCountAPI beginGatheringRequestCountTReX
#define endGatheringRequestCountAPI endGatheringRequestCountTReX
#define updateControllerAPI updateControllerTReX
#define initAPI initTReX
#define unloadSceneAPI unloadSceneTReX
#endif

#endif