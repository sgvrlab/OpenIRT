#include "stdafx.h"
#include <string>
#include <time.h>

#include "LoggerImplementationStdout.h"

using namespace std;

LoggerImplementationStdout::LoggerImplementationStdout(const char *name) {
	m_Name = name;
}


LoggerImplementationStdout::~LoggerImplementationStdout() {

}

void LoggerImplementationStdout::logMessage(LogLevel level, const char *message) {
	
	// output prefix depending on log level:
	switch (level) {
		case LOG_CRITICAL:	
			cout << "*CRITICAL*";
			break;
		case LOG_ERROR:
			cout << "ERROR";
			break;
		case LOG_WARNING:
			cout << "WARNING";
			break;
		case LOG_INFO:
			cout << "Info";
			break;
		case LOG_DEBUG:
			cout << "DEBUG";
			break;
		case LOG_DUMP:
			cout << "Dump";
			break;
		default:	
			cout << "-\t";
			break;
	}

	// output text:
	cout << " : " << message << std::endl;
}