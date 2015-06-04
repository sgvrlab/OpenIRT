#ifndef COMMON_LOGGERIMPLEMENTATIONSTDOUT_H
#define COMMON_LOGGERIMPLEMENTATIONSTDOUT_H

/********************************************************************
created:	2004/06/07
created:	7.6.2004   19:41
filename: 	Common\LoggerImplementationStdout.h
file base:	LoggerImplementationStdout
file ext:	h
author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)

purpose:	Logger that writes to standard output
*********************************************************************/

#include <iostream>
#include "Logger.h"

class LoggerImplementationStdout : public LoggerImplementation
{
public:
	LoggerImplementationStdout(const char *name = "main");
	~LoggerImplementationStdout();

	void logMessage(LogLevel level, const char *message);

protected:


private:
};


#endif