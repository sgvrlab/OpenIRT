#ifndef _VDR_TIMER_H_
#define _VDR_TIMER_H_

#include "systimer.h"
//#include <thread.h>
#include <stdio.h>

SysTimer g_systimer;
namespace Timers 
{
  int FRAME, CACHE_MISS, WS, AVG_TRAVERSE;


  //Semaphore *timerMutex = 0;
  // .. other timer IDs
  void setup()
  {
    //timerMutex = new Semaphore(1);
    {
#define ADD_CLOCK(c) c = g_systimer.addTimer( #c )
#define ADD_LOG(c) c = g_systimer.addDataLogger( #c )

      ADD_CLOCK ( FRAME );
    
	  ADD_LOG( CACHE_MISS );
	  ADD_LOG( WS );
	  ADD_LOG( AVG_TRAVERSE );



#undef ADD_CLOCK
#undef ADD_LOG
    }
    g_systimer.logAll(SysTimer::LOG_TO_MEM, 0);

  }
  void report()
  {
    FILE *fd;

#define DUMP(n) \
   fd = fopen( #n ".log", "wc"); \
   g_systimer.dump(Timers:: n, fd); \
   fclose(fd)

	 DUMP ( FRAME );
    
	  DUMP( CACHE_MISS );
	  DUMP( WS );
	  DUMP( AVG_TRAVERSE );



#undef DUMP
  }

  bool doTiming = false;
}
#define TICK(t) g_systimer.start(Timers:: t)
#define TOCK(t) g_systimer.stop(Timers:: t)
#define TICK_N(t,n) g_systimer.start(Timers:: t, n)
#define TOCK_N(t,n) g_systimer.stop(Timers:: t, n)
#define TICK_DAT(t,d) g_systimer.startWithData(Timers:: t, d)
#define TOCK_DAT(t,d) g_systimer.stopWithData(Timers:: t, d)
/*
#define TICK_SAFE(t) \
      do { Timers::timerMutex.down(); \
           g_systimer.start(Timers:: t); \
           Timers::timerMutex.up(); } while(0)
#define TOCK_SAFE(t) \
      do { Timers::timerMutex.down(); \
           g_systimer.stop(Timers:: t); \
           Timers::timerMutex.up(); } while(0)
*/
#define LOG_DAT(t,v) g_systimer.logData(Timers:: t,v);
#define LOG_DAT_N(t,v,n) g_systimer.logData(Timers:: t,v,n);

#endif
