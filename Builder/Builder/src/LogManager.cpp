#include "stdafx.h"

#include "LogManager.h"

// Logger implementations
#include "LoggerImplementationFileout.h"
#include "LoggerImplementationStdout.h"

using namespace std;

LogManager* LogManager::m_Singleton = 0;

LogManager::LogManager() {	
	// TODO: read options and create logs based on that

	// create a standard log	
	m_logFlags = LOG_INFO | LOG_WARNING | LOG_ERROR | LOG_CRITICAL | LOG_DEBUG;
	createLogger("main", LOG_STDOUT);
	setDefaultLogger("main");
}

LogManager::~LogManager() {
	// delete all created loggers
	LoggerList::iterator it;
	for (it = m_Loggers.begin(); it != m_Loggers.end(); ++it)
	{
		delete it->second;
	}
}

/**
* Log a text message using the given log level (and optional LogManager)
*/
void LogManager::logMessage(LogLevel level, const char *message, const char *LoggerName) {
	LoggerImplementation *logger;
	assert(m_defaultLogManager != 0);

	// do we log this log level ?
	if ((m_logFlags & level) != level)
		return;

	if (LoggerName != NULL) { // log with specified logger
		logger = m_Loggers[LoggerName];
		if (logger)
			logger->logMessage(level, message);
	}
	else
		m_defaultLogManager->logMessage(level, message);
}

/**
* Log a text message using the default log level (and optional LogManager)
*/
void LogManager::logMessage(const char *message, const char *LoggerName) {
	logMessage(LOG_INFO, message, LoggerName);
}

/**
* Set the default LogManager to the one specified 
*/	
bool LogManager::setDefaultLogger(const char *LoggerName) {
	if (m_Loggers.find(LoggerName) != m_Loggers.end())
		m_defaultLogManager = m_Loggers[LoggerName];
	else 
		return false; // not found

	// set to default, return OK
	return true;
}

/**
* Create a new LogManager with the specified name/handle
*/
bool LogManager::createLogger(const char *LoggerName, LogType logType) {
	LoggerImplementation *newLogger;

	// Test: Logger with that name already registered ?
	if (m_Loggers.find(LoggerName) != m_Loggers.end())
		return false;

	switch (logType) {
		case LOG_STDOUT:
			newLogger = new LoggerImplementationStdout(LoggerName);
			break;
		case LOG_FILETEXT:
			newLogger = new LoggerImplementationFileout(LoggerName);
			break;
		default:
			return false;
	}

	m_Loggers[LoggerName] = newLogger;

	if (!m_defaultLogManager)
		m_defaultLogManager = newLogger;

	return true;
}

bool LogManager::closeLogger(const char *LoggerName) {	
	if (m_Loggers.find(LoggerName) != m_Loggers.end()) {
		// Test: was this the default logger ?
		if (m_Loggers[LoggerName] == m_defaultLogManager) {
			LoggerList::iterator it = m_Loggers.begin();

			if (it->second == m_defaultLogManager) // same logger ? -> goto next
				it++;
			if (it == m_Loggers.end()) // there is no further logger left -> error
				return false; 
			
			m_defaultLogManager = it->second;
		}

		delete m_Loggers[LoggerName];
		m_Loggers.erase(LoggerName); // del from list
		return true;
	}
	else 
		return false; // not found
}

LogManager* LogManager::getSingletonPtr()
{
	if (m_Singleton == 0)
		m_Singleton = new LogManager();
	return m_Singleton;
}

LogManager& LogManager::getSingleton()
{  
	if (m_Singleton == 0)
		m_Singleton = new LogManager();
	return *m_Singleton;
}
