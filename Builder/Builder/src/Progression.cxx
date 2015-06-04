#include "Progression.h"
#include "Defines.h"

#include <stdio.h>

Progression::Progression(const char* process, float total, int count)
{
	init(process, total, count);
}

Progression::Progression()
{
	enumerations = -1;
}

void Progression::init(const char* process, float total, int count)
{
	if (count > 0) {
		printf("%s : ", process);
		percent      = 0;
		enumerations = 0;
		nextShown    = nextStep = (total-1.1f)/count;
		percentStep  = 100.0f/count;
	}
	else
		enumerations = -1;
}
/// next iteration
void Progression::step()
{
	if (enumerations < 0) return;

	if (++enumerations > nextShown) {
		nextShown += nextStep;
		percent   += percentStep;
		printf(" %.0f%%", percent);
		if (percent > 99.9f) {
			printf("\n");
		}
	}
}
