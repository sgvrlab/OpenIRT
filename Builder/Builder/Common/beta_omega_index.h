/********************************************************************************************************
// modified by bochang
// modified date : 2008/4/24
*********************************************************************************************************/
#ifndef _BETA_OMEGA_INDEX_H_
#define _BETA_OMEGA_INDEX_H_

#include "common.h"

class BetaOmegaIndex
{
public:
	vector< pair<int, int> > seq;

	BetaOmegaIndex(int s_x, int s_y, int width, int height) 
	{
		// we have constraints like this.
		if (width % 4 != 0 || height % 4 != 0 || width != height) 
		{
			printf("beta omega index error\n");
			return;
		}

		else
			make_sequence(0, 0, width);			

		//print_all_seq();
	}
	~BetaOmegaIndex() {};

private:	
	void beta_refinement(int s_x, 

	void make_sequence(int s_x, int s_y, int width)	
	{
		if (width == 1) 
			seq.push_back( make_pair(s_x, s_y) );
		else
		{
			int step = width>>1;
			make_sequence(s_x,		s_y,	  step);
			make_sequence(s_x+step, s_y,	  step);
			make_sequence(s_x,		s_y+step, step);
			make_sequence(s_x+step,	s_y+step, step);
		}
	}

	void print_all_seq()
	{
		for (int i=0; i<seq.size(); ++i)
			printf("(%d,%d)", seq[i].first, seq[i].second);
	}
};

#endif