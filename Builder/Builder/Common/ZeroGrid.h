#ifndef COMMON_ZEROGRID_H
#define COMMON_ZEROGRID_H

/********************************************************************
created:	2005/02/18
created:	18:2:2005   16:48
filename: 	c:\MSDev\MyProjects\Renderer\Common\ZeroGrid.h
file path:	c:\MSDev\MyProjects\Renderer\Common
file base:	ZeroGrid
file ext:	h
author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)

purpose:	Grid structure
*********************************************************************/

template <class T> class ZeroGrid
{
public:
	/**
	* Normal constructor, creates empty grid with values of
	* defaultValue for all grid points.
	*/
	ZeroGrid(unsigned int nx, unsigned int ny, unsigned int nz, T &defaultValue) {
		this->nx = nx;
		this->ny = ny;
		this->nz = nz;
		
		create();

		defaultElement = defaultValue;
	}

	ZeroGrid() {
		nx = 1;
		ny = 1;
		nz = 1;

		create();

		nxtimesny = nx*ny; // precalc offset
	}

	/**
	* Copy constructor
	*/
	ZeroGrid(const ZeroGrid &orig) {
		nx = orig.nx;
		ny = orig.ny;
		nz = orig.nz;
		nxtimesny = nx*ny; // precalc offset

		defaultElement = orig.defaultElement;

		m_Table = new T[nx*ny*nz];
		memcpy(m_Table, orig.m_Table, sizeof(T)*nx*ny*nz);
	}

	void operator=(const ZeroGrid &right) {
		nx = right.nx;
		ny = right.ny;
		nz = right.nz;
		nxtimesny = nx*ny; // precalc offset

		defaultElement = right.defaultElement;

		if (m_Table)
			delete m_Table;

		m_Table = new T[nx*ny*nz];
		memcpy(m_Table, orig.m_Table, sizeof(T)*nx*ny*nz);
	}

	/**
	* Dtor
	*/
	~ZeroGrid() {	
		clear();
	};

	/**
	* Change dimensions
	*/
	void setDimensions(unsigned int nx, unsigned int ny, unsigned int nz) {
		clear();

		this->nx = nx;
		this->ny = ny;
		this->nz = nz;

		create();
	}

	/**
	* Set default value if grid point not set yet
	*/
	void setDefaultValue(T &defaultValue) {
		defaultElement = defaultValue;
	}

	/**
	* Empties the hash table
	*/
	void clear() {		
		if (m_Table)
			delete m_Table;

		for (unsigned int i = 0; i < m_numBlocks; i++)
			if (m_BlockPointers[i] != 0)
				delete m_BlockPointers[i];

		if (m_BlockPointers)
			delete m_BlockPointers;
	}

	/**
	* Access operators:
	*/

	/**
	* Write access via bracket operator
	*/
	T &operator[](unsigned int address) { 
		assert(address < nx*ny*nz);
		return m_Table[address];
	}

	/**
	* Read access via bracket operator
	*/
	T &operator[](unsigned int address) const { 
		return m_Table[address];
	}

	/**
	* Read access via indicies
	*/
	T &at(unsigned int i, unsigned int j, unsigned int k) {
		return m_Table[k*nxtimesny + j*nx + i]; 
	}

	/**
	* Read access via offset
	*/
	T &at(unsigned int address) {
		return m_Table[address];
	}

	/**
	* Write access via indices
	*/
	void set(unsigned int i, unsigned int j, unsigned int k, T &value) {
		m_Table[k*nxtimesny + j*nx + i] = value;		
	}

	/**
	* Write access via offset
	*/
	void set(unsigned int address, T &value) {
		m_Table[address] =  value;
	}

	/**
	* Addition via indices
	*/
	void add(unsigned int i, unsigned int j, unsigned int k, T &value) {		
		m_Table[k*nxtimesny + j*nx + i] += value;
	}

	/**
	* Addition via offset
	*/
	void add(unsigned int address, T &value) {		
		m_Table[address] += value;
	}

	/**
	* Deletion via indices
	*/
	void erase(unsigned int i, unsigned int j, unsigned int k) {
		m_Table[k*nxtimesny + j*nx + i] = defaultElement;
	}

	/**
	* Deletion via offset
	*/
	void erase(unsigned int address) {
		m_Table[address] = defaultElement;
	}

	/**
	* Output statistics on the hash
	*/
	void printStats(const char *LoggerName = NULL) {
		LogManager *log = LogManager::getSingletonPtr();
		char outputBuffer[2000];
		sprintf(outputBuffer, "%d of %d blocks in used (%d%%)", m_numBlocksUsed, m_numBlocks, (int)((float)m_numBlocksUsed / (float)m_numBlocks)*100);
		log->logMessage(outputBuffer, LoggerName);
		sprintf(outputBuffer, "Mem usage :\t%d KB", (int)(m_numBlocksUsed*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*sizeof(T)/1024.0f));
		log->logMessage(outputBuffer, LoggerName);		
	}

protected:

	FORCEINLINE void checkAndSet(unsigned int i, unsigned int j, unsigned int k, T &value) {
		T* block = m_BlockPointers[k*bxtimesby + j*blockx + i];
		
		// not yet allocated ?
		if (block == 0) {
			block = new T[PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE];
			m_BlockPointers[k*bxtimesby + j*blockx + i] = block;
			memset(block, 0, sizeof(T)*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE*PHOTONGRID_BLOCKSIZE);
			m_numBlocksUsed++;
		}
		
		block[((k & (PHOTONGRID_BLOCKSIZE-1)) << (PHOTONGRID_BLOCKSIZEPOWER2+PHOTONGRID_BLOCKSIZEPOWER2))
		    + ((j & (PHOTONGRID_BLOCKSIZE-1)) << PHOTONGRID_BLOCKSIZEPOWER2)
		    + (i & (PHOTONGRID_BLOCKSIZE-1))] = value;
	}

	FORCEINLINE T &checkAndGet(unsigned int i, unsigned int j, unsigned int k) {
		T* block = m_BlockPointers[k*bxtimesby + j*blockx + i];

		// not yet allocated ?
		if (block == 0) {
			return defaultElement;
		}

		return block[((k & (PHOTONGRID_BLOCKSIZE-1)) << (PHOTONGRID_BLOCKSIZEPOWER2+PHOTONGRID_BLOCKSIZEPOWER2))
				   + ((j & (PHOTONGRID_BLOCKSIZE-1)) << PHOTONGRID_BLOCKSIZEPOWER2)
			       + (i & (PHOTONGRID_BLOCKSIZE-1))];
	}

	void create() {

		// Precalc and set offsets:
		nxtimesny = nx*ny; 
		blockx = nx >> PHOTONGRID_BLOCKSIZEPOWER2;
		blocky = ny >> PHOTONGRID_BLOCKSIZEPOWER2; 
		blockz = nz >> PHOTONGRID_BLOCKSIZEPOWER2;
		bxtimesby = blockx * blocky;

		// allocate tables:
		m_Table = new T[nx*ny*nz];
		memset(m_Table, 0, sizeof(T)*nx*ny*nz);

		m_numBlocks = blockx * blocky * blockz;
		m_numBlocksUsed = 0;

		m_BlockPointers = new T*[m_numBlocks];
		memset(m_BlockPointers, 0, sizeof(T*)*m_numBlocks);
	}

	// Hash table instance
	T *m_Table;

	// Array of pointers to blocks
	T **m_BlockPointers;

	// dimensions:
	unsigned int nx, ny, nz;
	unsigned int blockx, blocky, blockz;
	unsigned int nxtimesny, bxtimesby;

	unsigned int m_numBlocks, m_numBlocksUsed;

	// default value for grid points that have no value yet
	T defaultElement;

private:
};

#endif