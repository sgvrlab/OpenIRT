#ifndef COMMON_LOGMANAGER_H
#define COMMON_LOGMANAGER_H

struct eqstrLogManager
{
	bool operator()(const char* s1, const char* s2) const
	{
		return strcmp(s1, s2) == 0;
	}
};

struct greater_strLogManager {
   bool operator()(const char* x, const char* y) const {
      if ( strcmp(x, y) < 0)
         return true;

      return false;
   }
};


#include "Logger.h"
#include <hash_map>
#include <assert.h>

class LogManager
{
public:

	/**
	 * Log a text message using the given log level (and optional LogManager)
	 */
	void logMessage(LogLevel level, const char *message, const char *LoggerName = NULL);

	/**
	 * Log a text message using the default log level (and optional LogManager)
	 */
	void logMessage(const char *message, const char *LoggerName = NULL);

	/**
	 * Set the default LogManager to the one specified 
	 */	
	bool setDefaultLogger(const char *LoggerName);

	/**
	 * Create a new LogManager with the specified name/handle
	 */
	bool createLogger(const char *LoggerName, LogType logType = LOG_FILETEXT);

	/**
	 * Close a logger.
	 */
	bool closeLogger(const char *LoggerName);

	// get Singleton instance or create it
	static LogManager& getSingleton();
	static LogManager* getSingletonPtr();

protected:

	// Singleton, Ctor + Dtor protected
	LogManager();
	~LogManager();

	// Singleton instance
	static LogManager* m_Singleton;

	/// Determines which LogLevels will be written and which will be dropped (bit mask).
	/// See definition of LogLevel for all levels.
	unsigned int m_logFlags;
	
	/// LogManager we are currently using as default (implements custom way of
	/// writing logs, such as text files, HTML files, ...)
	LoggerImplementation *m_defaultLogManager;

	/// A list of all the logs the manager can access
	typedef stdext::hash_map<const char *, LoggerImplementation*, stdext::hash_compare<const char*, greater_strLogManager> > LoggerList;
	LoggerList m_Loggers;
	
private:
};

#endif