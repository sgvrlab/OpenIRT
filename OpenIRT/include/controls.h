#pragma once

namespace RendererType
{
	typedef enum Type
	{
		NONE,
		CPU_RAY_TRACER,
		CUDA_RAY_TRACER,
		CUDA_PATH_TRACER,
		CUDA_PHOTON_MAPPING,
		TREX,
		SIMPLE_RASTERIZER,		// use opengl starting from here
		DEBUGGING,
	};
};

typedef struct Controller_t
{
	enum FilterType
	{
		NONE,
		BOX,
		G_BUFFER
	};

	enum TileOrderingType
	{
		RANDOM,
		ROW_BY_ROW,
		Z_CURVE,
		HIGHEST_SALIENT_TILE_FIRST,
		CURSOR_GUIDED
	};

	bool useZCurveOrdering;
	bool shadeLocalIllumination;
	bool useShadowRays;
	bool gatherPhotons;
	bool showLights;
	bool useAmbientOcclusion;
	bool printLog;
	bool drawBackground;
	int pathLength;
	int numShadowRays;
	int numGatheringRays;
	int threadBlockSize;
	float timeLimit;
	int tileSize;
	FilterType filterType;
	int filterWindowSize;
	int filterIteration;
	float filterParam1;
	float filterParam2;
	float filterParam3;
	TileOrderingType tileOrderingType;
	int sizeMBForOOCVoxel;
	int warpSizeS;
	int warpSizeG;
	float AODistance;
	float envMapWeight;
	float envColWeight;

	Controller_t() : useZCurveOrdering(0), shadeLocalIllumination(1), useShadowRays(1), gatherPhotons(1), showLights(0), useAmbientOcclusion(0), printLog(1),
		pathLength(1), numShadowRays(1), numGatheringRays(0), threadBlockSize(256*64), timeLimit(30.0f), tileSize(32),
		filterType(G_BUFFER), filterWindowSize(10), filterIteration(3), filterParam1(0.5f), filterParam2(0.1f), filterParam3(0.1f), AODistance(20.0f),
		tileOrderingType(RANDOM)
		, sizeMBForOOCVoxel(512)
		, warpSizeS(1)
		, warpSizeG(1)
		, drawBackground(1)
		, envMapWeight(0.4f)
		, envColWeight(0.0f)
	{}
} Controller;

class Progress
{
	static void nullReset(int numSteps) {}
	static void nullStep() {}
	static void nullSetText(const char *text) {}

public:
	void (*reset)(int numSteps);
	void (*step)();
	void (*setText)(const char *text);

public:
	Progress(
		void (*reset)(int numSteps) = nullReset, 
		void (*step)() = nullStep, 
		void (*setText)(const char *text) = nullSetText)
		: reset(reset), step(step), setText(setText) {}
};

typedef void (*RenderDoneCallBack)(void *data);
