#ifndef COMMON_LOGGERIMPLEMENTATIONFILEOUT_H
#define COMMON_LOGGERIMPLEMENTATIONFILEOUT_H

/********************************************************************
	created:	2004/06/07
	created:	7.6.2004   19:41
	filename: 	Common\LoggerImplementationFileout.h
	file base:	LoggerImplementationFileout
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Logger that writes to log files.
*********************************************************************/

#include <fstream>
#include "Logger.h"

class LoggerImplementationFileout : public LoggerImplementation
{
public:
	LoggerImplementationFileout(const char *name = "main");
	~LoggerImplementationFileout();

	void logMessage(LogLevel level, const char *message);

protected:

	/// Log file we are outputting text into:
	std::ofstream m_outFile;
	
private:
};


#endif