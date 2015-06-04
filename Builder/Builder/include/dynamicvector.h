/*
===============================================================================

  FILE:  dynamicvector.h

  CONTENTS:

    the dynamicvector class allows adding and deleting of elements in constant
    time while keeping the n element indexed though numbers 0 to n-1. as the
    index of an element may change over time, the elements are expected to
    provide a field in which their index can be updated. therefore elements are
    only stored by references (e.g. not by value) in the dynamicvector. therefore
    the dynamicvector exclusively operates on void* pointers. 

  PROGRAMMERS:

    martin isenburg@cs.unc.edu

  COPYRIGHT:

    copyright (C) 2003  martin isenburg@cs.unc.edu

    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  CHANGE HISTORY:

    17 December 2003 -- added efficient handling for first (e.g. oldest) element
    03 October 2003 -- initial version created the day Peter had a sore throat

===============================================================================
*/
#ifndef DYNAMIC_VECTOR_H
#define DYNAMIC_VECTOR_H


// sungeui start --------------------------
const int VERTEX_IN_OTHER_CLUSTER = -1;
// sungeui end ----------------------------


class DynamicVector
{
public:
  DynamicVector();
  ~DynamicVector();

  int size() const;
  void* getFirstElement() const;
  void* getAndRemoveFirstElement();
  void* getElementWithRelativeIndex(int ri) const;
  int getRelativeIndex(void* d) const;
  void addElement(void* d);
  void removeElement(void* d);
  void removeFirstElement();

  // sungeui start ------------
  bool InitiateForNewCluster (bool DeleteVertexData, int & buffer_size, void ** buffer_address);
  bool DoesContain (void* d);
  // sungeui end ---------------

private:  
//  const int m_Which;
  void** data;
  int current_capacity;
  int current_capacity_mask;
  int current_begin;
  int current_end;
  int current_size;
};

#endif
