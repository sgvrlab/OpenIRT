#include "stdafx.h"
#include <string>
#include <time.h>

#include "LoggerImplementationFileout.h"

LoggerImplementationFileout::LoggerImplementationFileout(const char *name) {
	char currentTime[100];
	struct tm *newtime;
	time_t aclock;	
	std::string filename;
	m_Name = name;
	filename = m_Name;
	filename.append(".log");

	m_outFile.open(filename.c_str(), std::ios_base::out );

	time( &aclock );				  // Get time in seconds
	newtime = localtime( &aclock );   // Convert time to struct tm form

	strftime(currentTime, 100, "%d.%m.%Y %H:%M:%S", newtime);

	// add startup time
	m_outFile << "-----------------------------------------------------------\n"
		      << "Log '" << filename << "' starting at " << currentTime
			  << "\n-----------------------------------------------------------\n";
}

	
LoggerImplementationFileout::~LoggerImplementationFileout() {
	char currentTime[100];
	struct tm *newtime;
	time_t aclock;	

	time( &aclock );				  // Get time in seconds
	newtime = localtime( &aclock );   // Convert time to struct tm form

	strftime(currentTime, 100, "%d.%m.%Y %H:%M:%S", newtime);

	// add shutdown time
	m_outFile << "-----------------------------------------------------------\n"
		      << "Log stopping at " << currentTime
			  << "\n-----------------------------------------------------------\n";

	// close log file:
	if (m_outFile.is_open())
		m_outFile.close();
}

void LoggerImplementationFileout::logMessage(LogLevel level, const char *message) {
	char currentTime[100];
	struct tm *newtime;
	time_t aclock;	

	time( &aclock );				  // Get time in seconds
	newtime = localtime( &aclock );   // Convert time to struct tm form
	
	strftime(currentTime, 100, "%H:%M:%S", newtime);

	m_outFile << "[" << currentTime << " | ";

	// output prefix depending on log level:
	switch (level) {
		case LOG_CRITICAL:	
			m_outFile << "*CRITICAL*";
			break;
		case LOG_ERROR:
			m_outFile << "ERROR";
			break;
		case LOG_WARNING:
			m_outFile << "WARNING";
			break;
		case LOG_INFO:
			m_outFile << "Info";
			break;
		case LOG_DEBUG:
			m_outFile << "DEBUG";
			break;
		case LOG_DUMP:
			m_outFile << "Dump";
			break;
		default:	
			m_outFile << "-\t";
			break;
	}

	// output text:
	m_outFile << " ]\t" << message << std::endl;
}