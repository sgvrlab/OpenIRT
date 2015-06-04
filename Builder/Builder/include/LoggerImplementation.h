#ifndef COMMON_LOGGERIMPLEMENTATION_H
#define COMMON_LOGGERIMPLEMENTATION_H

#include <string>
#include "Logger.h"

/********************************************************************
	created:	2004/06/07
	created:	7.6.2004   19:45
	filename: 	c:\MSDev\MyProjects\Renderer\Common\LoggerImplementation.h
	file base:	LoggerImplementation
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Abstract base class for all classes implementing a 
	            log method for the log manager.
*********************************************************************/

class LoggerImplementation
{
public:
	LoggerImplementation(const char *name = "default") {
		m_Name = name;
	}

	virtual ~LoggerImplementation() {}
	
	virtual void logMessage(LogLevel level, const char *message) = 0;

protected:
	std::string m_Name;
	
private:
};

#endif