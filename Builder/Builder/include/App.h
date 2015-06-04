#define SimplifyRatio		0.05		// simplifing ratio within a cluster
//#define USE_HIGHER_PRECISION			// to allow have many vertices in PM
//#define	LAYOUT_HIERARCHY				// to get simplifcation info for layout computation

//#define DEBUG_MODE	
//#d\efine VIEWER_MODE
//#define DE_MODEL			// for finer partitioning
//#define DOE_MODEL			
#define SCAN_MODEL			// for lucy.


#define COLOR_INCLUDED		// if ply file has vertex color
#define NORMAL_INCLUDED		// if ply file has vertex normal
#define TEXTURE_INCLUDED	// if ply file has vertex texture


#define  _TIMER_OUTPUT_	
//#define DISABLE_OCCLUSION
//#define SUB_OBJECT			// for finer granuality of occlusion culling
#define LAYOUT
//#define Z_ORDER		// ordering for PM



//#define HOPPE		// to compute HOPPE's layout

//#define MATTHEW_DEMO		// load necessary files and disable prefetch
//#define NO_VIEWER


#define EXTRAFAST           // faster RC of martin
//#define PETER_RCODER        // enable/disable Peter's Range coder