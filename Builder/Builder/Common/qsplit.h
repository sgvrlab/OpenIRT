#ifndef _QSPLIT_H_
#define _QSPLIT_H_

int qsplit(ModelInstance** pObjectList, int obj_size, float pivot_val, int axis) 
{
	BBox bbox;
	double centroid;
	int ret_val = 0;

	for (int i = 0; i < obj_size; ++i) {
		bbox.setValue(pObjectList[i]->bb);
		centroid = ((bbox.getmin())[axis] + (bbox.getmax())[axis]) / 2.0f;
		if (centroid < pivot_val) {
			ModelInstance* temp = pObjectList[i];
			pObjectList[i]		= pObjectList[ret_val];
			pObjectList[ret_val] = temp;
			++ret_val;
		}
	}
	if (ret_val == 0 || ret_val == obj_size) 
		ret_val = obj_size/2;

	return ret_val;
}

#endif