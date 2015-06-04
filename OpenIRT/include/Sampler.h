#pragma once

#include "Vector2.h"
#include "random.h"
#include <vector>
#include <stopwatch.h>

namespace irt
{

class Sampler
{
protected:
	int m_numPixelX, m_numPixelY;
	int m_numSubSample;
	Vector2 *m_samplerPool;

public:
	Sampler() : m_numPixelX(0), m_numPixelY(0), m_numSubSample(0), m_samplerPool(0) {}

	~Sampler()
	{
		clear();
	}

	void clear()
	{
		if(m_samplerPool)
			delete[] m_samplerPool;
		m_samplerPool = 0;
	}

	void reset(int numPixelX, int numPixelY, int numSubSample)
	{
		clear();

		m_numPixelX = numPixelX;
		m_numPixelY = numPixelY;
		m_numSubSample = numSubSample;

		m_samplerPool = new Vector2[numPixelX*numPixelY*numSubSample];
	}

	void resample(int frame)
	{
		unsigned int seed = tea<16>((unsigned int)frame, 0);

		static int timer = StopWatch::create();
		StopWatch::get(timer).start();

		// jittering
		int subSize = (int)sqrtf((float)m_numSubSample+0.1f);
		std::vector<int> IDs;

		float delta = 1.0f / subSize;
		for(int y=0;y<m_numPixelY;y++)
		{
			for(int x=0;x<m_numPixelX;x++)
			{
				for(int i=0;i<m_numSubSample;i++)
					IDs.push_back(i);

				for(int i=0;i<m_numSubSample;i++)
				{
					Vector2 &sample = m_samplerPool[i+(x+y*m_numPixelX)*m_numSubSample];

					int pos = (int)(rnd(seed) * IDs.size());
					int ID = IDs[pos];
					IDs[pos] = IDs[IDs.size()-1];
					IDs.pop_back();

					int sx = ID % subSize;
					int sy = ID / subSize;

					sample.e[0] = ((float)sx + rnd(seed)) * delta;
					sample.e[1] = ((float)sy + rnd(seed)) * delta;
				}
			}
		}

		StopWatch::get(timer).stop();
		//printf("Sample time: %f ms\n", StopWatch::get(timer).getTime());
		StopWatch::get(timer).reset();
	}

	const Vector2 &getSample(int x, int y, int frame)
	{
		return m_samplerPool[(frame%m_numSubSample)+(x+y*m_numPixelX)*m_numSubSample];
	}

	int getNumSubSample() {return m_numSubSample;}
};

};