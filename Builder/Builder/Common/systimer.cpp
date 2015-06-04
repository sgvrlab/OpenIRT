/******************************************************************************
 * SysTimer.cpp  - Gather and report system timing information
 *
 *****************************************************************************/

#include "stdafx.h"

#include "systimer.h"
#include <assert.h>


//----------------------------------------------------------------------------
SysTimer::SysTimer()
{
  m_globalEnable = true;
}

//----------------------------------------------------------------------------
SysTimer::~SysTimer()
{
  
}

//----------------------------------------------------------------------------
int SysTimer::addTimer(const char *name)
{
  int id  = -1;
  // find free slot
  TimerSet::iterator ti = m_timers.begin();
  for (; ti != m_timers.end(); ++ti) {
    if (ti->type == TimerT::UNUSED) {
      break; // found an 'empty'
    }
  }
  if (ti != m_timers.end()) {
    // found an empty slot
    ti->timer = new Stopwatch(name);
    id = ti - m_timers.begin();
  }
  else {
    // didn't find a slot
    id = m_timers.size();
    m_timers.push_back( TimerT( new Stopwatch(name) ) );
  }
  return id;
}

//----------------------------------------------------------------------------
int SysTimer::addDataLogger(const char *name)
{
  int id  = -1;
  // find free slot
  TimerSet::iterator ti = m_timers.begin();
  for (; ti != m_timers.end(); ++ti) {
    if (ti->type == TimerT::UNUSED) {
      break; // found an 'empty'
    }
  }
  if (ti != m_timers.end()) {
    // found an empty slot
    ti->timer = 0;
    ti->type = TimerT::LOGGER;
    id = ti - m_timers.begin();
  }
  else {
    // didn't find a slot
    id = m_timers.size();
    m_timers.push_back( TimerT() );
    m_timers.rbegin()->type = TimerT::LOGGER;
  }
  return id;
}


//----------------------------------------------------------------------------
void SysTimer::removeTimer(int timer_id)
{
  if (timer_id >= m_timers.size() ) {
    // cerr out of bounds!
    return;
  }
  
  TimerT &t = m_timers[timer_id];
  Stopwatch *swatch = t.timer;
  delete swatch;
  t.timer = 0;
  t.type = TimerT::UNUSED;

  // trim out trailing zero-timers
  if (timer_id == m_timers.size() - 1) {
    while (m_timers.size()>0 && m_timers[ timer_id-- ].timer == 0)
      m_timers.pop_back();
  }
}

//----------------------------------------------------------------------------
void SysTimer::startAll()
{
  TimerSet::iterator ti = m_timers.begin();
  for (; ti != m_timers.end(); ++ti) {
    if (ti->timer) ti->timer->Start();
  }
}

//----------------------------------------------------------------------------
void SysTimer::stopAll()
{
  TimerSet::iterator ti = m_timers.begin();
  for (; ti != m_timers.end(); ++ti) {
    if (ti->timer) ti->timer->Stop();
  }
}

//----------------------------------------------------------------------------
void SysTimer::resetAll()
{
  TimerSet::iterator ti = m_timers.begin();
  for (; ti != m_timers.end(); ++ti) {
    if (ti->timer) ti->timer->Reset();
  }
}



//----------------------------------------------------------------------------
void SysTimer::log(int timer_id, unsigned int logMode, FILE* fd)
{
  m_timers[timer_id].logMask |= (1<<logMode);
  if (logMode == LOG_TO_FD1) {
    assert(fd!=0);
    m_timers[timer_id].fd1 = fd;
  }
  else if (logMode == LOG_TO_FD2) {
    assert(fd!=0);
    m_timers[timer_id].fd2 = fd;
  }
}
//----------------------------------------------------------------------------
void SysTimer::unlog(int timer_id, unsigned int logMode)
{
  m_timers[timer_id].logMask &= ~(1<<logMode);
}
//----------------------------------------------------------------------------
void SysTimer::logAll(unsigned int logMode, FILE *fd)
{
  TimerSet::iterator ti = m_timers.begin();
  for (; ti != m_timers.end(); ++ti) 
  {
    if (ti->type == TimerT::UNUSED) continue;
    ti->logMask |= (1<<logMode);
    if (logMode == LOG_TO_FD1) {
      assert(fd!=0);
      ti->fd1 = fd;
    }
    else if (logMode == LOG_TO_FD2) {
      assert(fd!=0);
      ti->fd2 = fd;
    }
  }
}
//----------------------------------------------------------------------------
void SysTimer::unlogAll(unsigned int logMode, FILE *fd)

{
  TimerSet::iterator ti = m_timers.begin();
  for (; ti != m_timers.end(); ++ti) {
    if (ti->type != TimerT::UNUSED) ti->logMask &= ~(1<<logMode);
  }
}



//----------------------------------------------------------------------------
void SysTimer::dump(int timer_id, FILE *fd)
{
  // destructive dump
  TimerT &t = m_timers[timer_id];
  while (!t.log.empty()) {
    LogEntry &l = t.log.front();
    fprintf(fd, "%d %d %d %d %d %f\n",
            l.timerID, l.evNumber, l.type, l.multiplicity, l.dataValue, l.deltaT);
    t.log.pop();
  }
}

//----------------------------------------------------------------------------
void SysTimer::dumpAll(FILE *fd)
{
  TimerSet::iterator ti = m_timers.begin();
  for (int id=0; ti != m_timers.end(); ++ti, ++id) {
    if (ti->timer) 
      dump(id, fd);
  }
}

//----------------------------------------------------------------------------
void SysTimer::enableAll()
{
  m_globalEnable = true;
}

//----------------------------------------------------------------------------
void SysTimer::disableAll()
{
  m_globalEnable = false;
}



//----------------------------------------------------------------------------
void SysTimer::setFilter(
  int timer_id, unsigned int Mode, void* param1, void* param2)
{
/*
  switch (Mode) {
    case BOX_FILTER:
      t.msec = param1;
      t.nboxsamp = param2;
      break;
    case FIR_FILTER:
      t.nfirsamp = param1;
      t.samp = param2;
      break;
    case IIR_FILTER:
      t.weight = param1;
      break;
    default:
      // cerr << you ninny! << endl;
  }
*/  
}

//----------------------------------------------------------------------------
void SysTimer::getFilter(
  int timer_id, unsigned int *Mode, void **param1, void **param2)
{
/*
  TimerT &t = m_timers[timer_id];

  *Mode = t.filterMode;
  switch(t.filtermode) {
    case BOX_FILTER:
      *param1 = (void*) t.msec;
      *param2 = (void*) t.nboxsamp;
      break;
    case FIR_FILTER:
      *param1 = (void*) t.nfirsamp;
      *param2 = (void*) t.samp;
      break;
    case IIR_FILTER:
      *param1 = (void*) t.weight;
      break;
    default:
      // cerr << you blew it! << endl;
  }
*/
}




//----------------------------------------------------------------------------
void _updateTimer(int timer_id)
{
/*
  // ITS ALL BOGUS!!!


  // This kind of thing is all for interactive display of averages.
  // For system diagnosis, we can just buffer up all the data and analyze it
  // off like with whatever kind of filtering desired.
  TimerT &t = m_timers[timer_id];
  switch(t.filterMode)
  {
    case BOX_FILTER:
      if (t.msec > 0 && t.nboxsamp > 0) {
        // average n samples or when at least msec elapses
        

      }
      else if (t.msec > 0) {
        // Always integrate over msec (or more)
      }
      else if (t.nboxsamp > 0) {
        // use the last n samples regardless of time
      }

      break;
    case FIR_FILTER:
      float sum = 0;
      if (history.size() == 0) {
        t.firsum = 0; 
      }
      else if (history.size() < t.nfirsamp)
      {
        t.firsum += t.samp[history.size()-1];
      }
      
      for (int i=0; i < t.nfirsamp ; i++) 
      {
        sum += t.samp[i] * history[i];
      }
      sum /= totalweight;
      break;
    case IIR_FILTER:
      avg *= 1.0f - t.weight;
      avg += new_period*t.weight;
      break;
  }
*/
}

//----------------------------------------------------------------------------
void SysTimer::_logEvent(int timer_id, unsigned char ev, int ev_id)
{
  TimerT &t = m_timers[timer_id];

  // make a new log entry
  LogEntry l;
  l.timerID = timer_id;
  l.evNumber = (ev_id >= 0) ? ev_id : t.timer->GetNumStarts();
  l.type = ev;
  l.dataValue = 0;
  l.multiplicity = 1;
  l.deltaT = t.timer->GetTime();

  if (t.logMask & (1<<LOG_TO_MEM)) {
    t.log.push(l);
  }
  if (t.logMask & (1<<LOG_TO_FD1)) {
    fprintf(t.fd1, "%d %d %d %d %f\n",
            l.timerID, l.evNumber, l.type, l.multiplicity, l.deltaT);
  }
  if (t.logMask & (1<<LOG_TO_FD2)) {
    fprintf(t.fd2, "%d %d %d %d %f\n",
            l.timerID, l.evNumber, l.type, l.multiplicity, l.deltaT);
  }
}

//----------------------------------------------------------------------------
void SysTimer::_logDataEvent(int timer_id, unsigned char ev, int data, int ev_id )
{
  TimerT &t = m_timers[timer_id];

  // make a new log entry
  LogEntry l;
  l.timerID = timer_id;
  l.evNumber = (ev_id >= 0) ? ev_id : (t.timer ? t.timer->GetNumStarts() : 0);
  l.type = ev;
  l.dataValue = data;
  l.multiplicity = 1;
  l.deltaT = t.timer ? t.timer->GetTime() : 0;

  if (t.logMask & (1<<LOG_TO_MEM)) {
    t.log.push(l);
  }
  if (t.logMask & (1<<LOG_TO_FD1)) {
    fprintf(t.fd1, "%d %d %d %d %d %f\n",
            l.timerID, l.evNumber, l.type, l.multiplicity, l.dataValue, l.deltaT);
  }
  if (t.logMask & (1<<LOG_TO_FD2)) {
    fprintf(t.fd2, "%d %d %d %d %d %f\n",
            l.timerID, l.evNumber, l.type, l.multiplicity, l.dataValue, l.deltaT);
  }
}
