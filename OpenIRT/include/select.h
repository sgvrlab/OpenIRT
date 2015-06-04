#ifndef SELECT_H
#define SELECT_H

#ifdef ElemIndex

template<class Elem> inline void swap(Elem* list, int a, int b)
{
  Elem temp = list[a];
  list[a] = list[b];
  list[b] = temp;
}

template<class Elem> inline int mid_component(Elem a[3])
{
  return ((a[0] > a[1]) ? ((a[0] < a[2]) ? 0 : ((a[1] > a[2]) ? 1 : 2)) : \
	            ((a[1] < a[2]) ? 1 : ((a[0] > a[2]) ? 0 : 2)));
}

template<class Elem, int axis>
inline int mid_component(Elem a[3])
{
  return ((ElemIndex(a[0],axis) > ElemIndex(a[1],axis)) ? ((ElemIndex(a[0],axis) < ElemIndex(a[2],axis)) ? 0 : ((ElemIndex(a[1],axis) > ElemIndex(a[2],axis)) ? 1 : 2)) : \
          ((ElemIndex(a[1],axis) < ElemIndex(a[2],axis)) ? 1 : ((ElemIndex(a[0],axis) > ElemIndex(a[2],axis)) ? 0 : 2)));
}

/*
  function partition(list, left, right, pivotIndex)
     pivotValue := list[pivotIndex]
     swap list[pivotIndex] and list[right]  // Move pivot to end
     storeIndex := left
     for i from left to right-1
         if list[i] < pivotValue
             swap list[storeIndex] and list[i]
             storeIndex := storeIndex + 1
     swap list[right] and list[storeIndex]  // Move pivot to its final place
     return storeIndex
*/

/*
  returns index such
  list[left]..list[index-1] < list[index] < list[index+1]..list[right]

  or list[pivotIndex] is moved to list[index] where the item would be if the
  entire list were sorted.
 */
template<class Elem, int axis> int partitionOld(Elem* list, int left, int right, int pivotIndex)
{
  Elem pivotValue = list[pivotIndex];
  swap(list, pivotIndex, right); // Move pivot to end
  int storeIndex = left;
  for(int i = left; i <= right-1; ++i) {
    if (ElemIndex(list[i],axis) < ElemIndex(pivotValue,axis)) {
      swap(list, storeIndex, i);
      storeIndex = storeIndex + 1;
    }
  }
  swap(list, right, storeIndex);  // Move pivot to its final place
  return storeIndex;
}

// The original partition algorithm I took this from only garanteed that the elements on
// the left were < pivotValue and the elements on the right were >= pivotValue.
// Unfortunately it didn't garantee that the pivotValue was in its sorted position.  By
// swapping out the pivot value before partitioning, and then swapping back in at the
// partition split, we can now garantee that the index returned is in its sorted position.
template<class Elem, int axis> int partition(Elem* list, int left, int right, int pivotIndex)
{
  Elem pivotValue = list[pivotIndex];
  //cout << "pivot = "<<pivotValue<<" ";
  swap(list, right, pivotIndex);
  pivotIndex = right;
  left--;
  while(1) {
    do {
      left++;
    } while(left < right && ElemIndex(list[left],axis)  < ElemIndex(pivotValue,axis));
    do {
      right--;
    } while(left < right && ElemIndex(list[right],axis) > ElemIndex(pivotValue,axis));
    if ( left < right ) {
      swap(list, left, right);
    } else {
      // Put the pivotValue back in place
      swap(list, left, pivotIndex);
      return left;
    }
  }
}

/*
   function select(list, left, right, k)
     loop
         select pivotIndex between left and right
         pivotNewIndex := partition(list, left, right, pivotIndex)
         if k = pivotNewIndex
             return list[k]
         else if k < pivotNewIndex
             right := pivotNewIndex-1
         else
             left := pivotNewIndex+1

*/

/*
  returns the kth largest value in the list.  A side effect is that
  list[left]..list[k-1] < list[k] < list[k+1]..list[right].
*/

template<class Elem, int axis> Elem select(Elem* list, int left, int right, int k)
{
  int numIters = 0;
  while(1) {
    ++numIters;
    // select a value to pivot around between left and right and store the index to it.
    int pivotIndex = (left+right)/2;
    // Determine where this value ended up.
    int pivotNewIndex = partition<Elem,axis>(list, left, right, pivotIndex);
    if (k == pivotNewIndex) {
      // We found the kth value
      //std::cout << "numIters = "<<numIters<<"\n";
      return list[k];
    } else if (k < pivotNewIndex)
      // if instead we found the k+Nth value, remove the segment of the list
      // from pivotNewIndex onward from the search.
      right = pivotNewIndex-1;
    else
      // We found the k-Nth value, remove the segment of the list from
      // pivotNewIndex and below from the search.
      left = pivotNewIndex+1;
  }
}

/*
  right is inclusive
  */
template<class Elem, int axis> Elem selectSmart(Elem* list, int left, int right, int k)
{
  int numIters = 0;
  while(1) {
    ++numIters;
    if(right-left <=5) {
/*
procedure bubbleSort( A : list of sortable items ) defined as:
  n := length( A )
  do
    swapped := false
    n := n - 1
    for each i in 0 to n - 1  inclusive do:
      if A[ i ] > A[ i + 1 ] then
        swap( A[ i ], A[ i + 1 ] )
        swapped := true
      end if
    end for
  while swapped
end procedure
*/

      int n = right-left+1;
      bool swapped;
      do {
        --n;
        swapped = false;
        for(int i = left; i < n+left; ++i)
          if (ElemIndex(list[i+1],axis) < ElemIndex(list[i],axis)) {
            swap(list, i, i+1);
            swapped = true;
          }
      } while (swapped);
      return list[k];
    }

    // select a value to pivot around between left and right and store the index to it.
    int ps[3];
    for(int i=0;i<3;++i)
      ps[i] = static_cast<int>(static_cast<double>(rand()) / (RAND_MAX) * (right - left)) + left;
    Elem ps_val[3];
    for(int i=0;i<3;++i)
      ps_val[i] = list[ps[i]];
    //int pivotIndex = ps[mid_component<int, Comp<int, 0> >(ps_val)];
    int pivotIndex = ps[mid_component<Elem, axis>(ps_val)];
    // Determine where this value ended up.
    int pivotNewIndex = partition<Elem,axis>(list, left, right, pivotIndex);
    if (k == pivotNewIndex) {
      // We found the kth value
      //std::cout << "numIters = "<<numIters<<"\n";
      return list[k];
    } else if (k < pivotNewIndex)
      // if instead we found the k+Nth value, remove the segment of the list
      // from pivotNewIndex onward from the search.
      right = pivotNewIndex-1;
    else
      // We found the k-Nth value, remove the segment of the list from
      // pivotNewIndex and below from the search.
      left = pivotNewIndex+1;
  }
}

#endif
#endif