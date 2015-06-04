#ifndef COMMON_VIEWER_H
#define COMMON_VIEWER_H

#include <vector>

/**
 * Encapsulates information about a camera/viewer
 * position (as, perhaps, imported from a scene file).
 */
typedef struct View_t {
	Ray view;
	char name[100];

	// TODO: camera intrinsics ? view parameters ?
} View;

typedef std::vector<View> ViewList;
typedef ViewList::iterator ViewListIterator;

#endif