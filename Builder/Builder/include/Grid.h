#ifndef GRIDHEADER_H
#define GRIDHEADER_H

#include "Box.h"
#include "Point.h"

/** a grid allowing for fast quantization */
struct Grid : public Box
{
	/// store the resolution in x, y and z direction
	Point<int> resolution;
	///
	int planeSize;
	///
	int size;
	///
	Vector3 scale;
	///
	Vector3 width;
	/// constructor
	Grid() 
	{
		size = 0;
	}
	/// set resolution from cell side width
	void setCellWidth(float cellWidth)
	{
		for (int c=0; c<3; ++c) 
		{
			resolution[c] = (int)( (p_max[c]-p_min[c])/cellWidth + 0.9999 );
			if(resolution[c] <= 0) resolution[c] = 1;
		}
		recalcHash();
	}	
	///
	void setResolution(const Point<int>& res) { resolution = res; recalcHash(); }
	/// set resolution, such that longest side is subdivided k times
	void setResolution(int k)
	{
		setCellWidth((p_max-p_min).maxComponent()/k);
	}
	/// return the surface area of the box
	int getSurfaceCells() const 
	{ 
		return (resolution[0]*(resolution[1]+resolution[2])+resolution[1]*resolution[2])*2;
	}
	/// compute values to faster compute the cell index
	void recalcHash()
	{
		planeSize = resolution[0]*resolution[1];
		size = resolution[2]*planeSize;
		// taejoon
		if(size == 0) size = 1;
		int c;
		for (c=0; c<3; ++c) scale[c] = resolution[c]/(p_max[c]-p_min[c]);
		for (c=0; c<3; ++c) width[c] = (p_max[c]-p_min[c])/resolution[c];
	}
	/// return the total number of cells
	int getSize() const { return size; }
	/// return the indices in x-, y- and z-direction
	Point<int> getCellIndices_pt(const Vector3& p) 
	{
		Point<int> i((int) (scale[0]*(p[0]-p_min[0])), 
			         (int) (scale[1]*(p[1]-p_min[1])), 
					 (int) (scale[2]*(p[2]-p_min[2])) );
		for (int c=0; c<3; ++c) if (i[c] >= resolution[c]) i[c] = resolution[c] - 1;
		for (int c=0; c<3; ++c) if (i[c] < 0) i[c] = 0;

		return i;
	}
	/// return the index of the cell in which p is
	int getCellIndex_(const Point<int>& i) const
	{
		return i[0]+i[1]*resolution[0]+i[2]*planeSize;
	}
	int getCellIndex(const Point<int>& i) const
	{
		return i[0]+i[1]*resolution[0]+i[2]*planeSize;
	}
	/// return the index of the cell in which p is
	int getCellIndex(const Vector3& p) 
	{
		return getCellIndex_(  getCellIndices_pt(p));
	}
	/// return the index point of a index
	Point<int> getCellIndices(int i) const
	{
		return Point<int>(i % resolution[0], (i%planeSize)/resolution[0], i/planeSize);
	}

	void getCellBoundingBox(int i, Vector3 &min, Vector3 &max) {
		Vector3 coord;
		Vector3 bb_min = Vector3(p_min[0], p_min[1], p_min[2]);		
		//i[0]+i[1]*resolution[0]+i[2]*planeSize;
		coord[2] = i / planeSize;
		int temp = i % planeSize;
		coord[1] = temp / resolution[0];
		coord[0] = temp % resolution[0];		
		min = bb_min + vmul(coord, width);
		max = bb_min + vmul((coord + Vector3(1,1,1)), width);
	}

	void getCellBoundingBox(const Point<int>& i, Vector3 &min, Vector3 &max) {
		Vector3 coord(i[0], i[1], i[2]);
		min = p_min + vmul(coord, width);
		max = min + width;
	}
};

#define SIZE_ADAPTIVE_VOXEL 108	// for 32bit <-> 64bit conversion
class AdaptiveVoxel
{
public:
	int cellIndex;
	int fileIndex;	// voxel file index, -1: empty voxel, -2: have child voxels
	int neighbor[6];
private:
	Grid grid;
public:
	AdaptiveVoxel *children;
	AdaptiveVoxel *parent;

public:
	AdaptiveVoxel() 
	{
		cellIndex = -1;
		fileIndex = -1;
		for(int i=0;i<6;i++) 
			neighbor[i] = -1;
		children = NULL;
		parent = NULL;
	}

	~AdaptiveVoxel()
	{
		if(children) delete[] children;
	}

	void setGrid(Grid *g)
	{
		memcpy(&grid, g, sizeof(Grid));
		if(children) delete[] children;
		children = new AdaptiveVoxel[grid.size];
	}

	Grid *getGrid()
	{
		return &grid;
	}

	int saveToFile(FILE *fp)
	{
		int num = fwrite(this, SIZE_ADAPTIVE_VOXEL, 1, fp);
		for(int i=0;i<grid.size;i++)
			num += children[i].saveToFile(fp);
		return num;
	}

	bool loadFromFile(FILE *fp)
	{
		if(children) delete[] children;
		bool success = fread(this, SIZE_ADAPTIVE_VOXEL, 1, fp) == 1;
		children = new AdaptiveVoxel[grid.size];
		for(int i=0;i<grid.size;i++)
		{
			success &= children[i].loadFromFile(fp);
			children[i].parent = this;
		}
		return success;
	}

	void arrangeFileIdx(stdext::hash_map<int, int> &map)
	{
		if(map.find(fileIndex) != map.end())
			fileIndex = map[fileIndex];
		for(int i=0;i<grid.size;i++)
			children[i].arrangeFileIdx(map);
	}
};


#endif