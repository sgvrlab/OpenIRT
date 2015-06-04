#ifndef PROGRESSION_H
#define PROGRESSION_H


/** progression provides a simple possibility to show progression of process in debug window */
class Progression
{
public:
	float nextShown;
	float nextStep;
	float percent;
	float percentStep;
	int   enumerations;
	/// create empty progression
	Progression();
	/// create from total enumerations and number of times to print progression
	Progression(const char* process, float total, int count);
	/// reinitialize
	void init(const char* process, float total, int count);
	/// next iteration
	void step();
};

#endif
