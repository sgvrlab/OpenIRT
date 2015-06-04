#ifndef COMMON_LOGGER_H
#define COMMON_LOGGER_H

/**
 * Include this into files that need logging.
 * 
 * This file includes all other necessary logger files and
 * defines constants etc.
 */

// Available categories for log messages
enum LogLevel {
	LOG_CRITICAL = 0x01, // critical error (mostly resulting in program termination)
	LOG_ERROR    = 0x02, // 'normal' error
	LOG_WARNING  = 0x04, // warning (possible error)
	LOG_INFO     = 0x08, // information (normal output)
	LOG_DEBUG    = 0x10, // debug information (more verbose output for developers)
	LOG_DUMP     = 0x20  // dump of large amount of debug information
};

// Types of loggers
enum LogType {
	LOG_STDOUT,
	LOG_FILETEXT,
	LOG_FILEHTML,
};

#include "LoggerImplementation.h"
#include "LogManager.h"

#endif