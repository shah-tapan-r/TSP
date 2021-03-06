// This is the implementation module of the template class Heap.
// It defines the methods for handling a 2-minheap with entries of the
// template type T. A method "decrease_value" is supported!

#ifndef HEAP_H
#define HEAP_H

#include <ARRAY/array.h>
#include <iostream>
 
using namespace std;


template <class T, class E>
class HHeap;

// Evaluater class E must provide operators < and <=!
template <class T, class E>
class HHeapNode
{
  T cont;
  E value;
  long pos;
public: 
 HHeapNode(void): pos(-1) {}
  HHeapNode(const T& c, E v, long p): 
    cont(c), value(v), pos(p) 
    {}
    HHeapNode<T,E>& operator=(const HHeapNode<T,E>& h)
    {
      cont = h.cont; 
      value = h.value; 
      pos = h.pos;
      return *this;
    }
  ~HHeapNode(void) {}
  bool operator<= (const HHeapNode<T,E>& r) { return value <= r.value; }
  bool operator<= (E r) { return value <= r; }
  bool operator< (const HHeapNode<T,E>& r) { return value < r.value; }
  bool operator< (E r) { return value < r; }
  bool operator>= (const HHeapNode<T,E>& r) { return !operator<(r);}
  bool operator>= (E r) { return !operator<(r);}
  bool operator> (const HHeapNode<T,E>& r) { return !operator<=(r);}
  bool operator> (E r) { return !operator<=(r);}
  bool operator== (const HHeapNode<T,E>& r) { return (operator<=(r))&&(operator>=(r));}
  bool operator== (E r) { return (operator<=(r))&&(operator>=(r));}

  void print(void)
    {
      cout << "Value " << value << ", Content " << cont << ", Position " << pos << endl;
    }

  friend class HHeap<T,E>;
};

// Evaluater class E must provide operators < and <=!
template <class T, class E>
class HHeap
{
  long size;
  Array<long> heap;
  Array<HHeapNode<T,E> > handleArray;
  long firstFreeHandle; 

  long dad(long i) { return ((i-1)>>1);};
  long leftson(long i) { return (i<<1)+1;};
  long minimum(long first, long last);
  long minimum(long first);
  void pull(long a, long b);
  void up(long i, long handle, long root=0);
  void down_botup(long i, long handle);

public:
  HHeap(long m);
  HHeap(const HHeap<T,E>& h);
  HHeap<T,E>& operator=(const HHeap<T,E>& h); 
  ~HHeap(void) {};
  void clear(void);
  long ins(const T& c, E val, bool heapi=true);
  void heapify(void);
  long del(void);  
  long del(long handle); // NOT TESTED!!!
  void decrease_value(long handle, E value);
  void increase_min(E value);
  bool empty(void);
  long getSize(void) { return size;};
  bool min(E& v, T& c);
  T min(E& v);
  T min(void);
  E minValue(T& c);
  E minValue(void);
  bool ass(void);
  void print(void);
};


//private:
template <class T, class E>
inline HHeap<T,E>::HHeap(const HHeap<T,E>& h) : 
size(h.size), heap(h.heap), handleArray(h.handleArray), firstFreeHandle(h.firstFreeHandle)
{}

template <class T, class E>
inline HHeap<T,E>& HHeap<T,E>::operator=(const HHeap<T,E>& h) 
{
  if (this==&h) return *this;
  size = h.size;
  heap = h.heap; 
  handleArray = h.handleArray;
  firstFreeHandle = h.firstFreeHandle;
  return *this;
}



template <class T, class E>
inline long HHeap<T,E>::minimum(long first, long last)

     // return the index of the minimum of the elements in heap
     // between including first and last

{
  assert((last>=first)&&(first>=0));
  long i;
  if (size<=last) last = size-1;
  long min;
  min = first;
  for (i=last; i>first; i--)
    {
      if (handleArray[heap[i]]<handleArray[heap[min]])
	min=i;
    }
  return min;
}

template <class T, class E>
inline long HHeap<T,E>::minimum(long first)

     // return the index of the minimum of the elements in heap
     // between including first and first+1

{
  assert(first>=0);
   long min;
  if (size-1==first) return first;
  if (handleArray[heap[first]]<handleArray[heap[first+1]])
    min=first;
  else
    min=first+1;
  return min;
}

template <class T, class E>
inline void HHeap<T,E>::pull(long a, long b)
{
  handleArray[heap[b]].pos=a;
  heap[a]=heap[b];
}


template <class T, class E>
inline void HHeap<T,E>::up(long i, long handle, long root)
     // does the up-heap; i is the index, where the new element elem is inserted
{
  long father;
  while(i>root)
    { 
      father = dad(i);
      if (handleArray[heap[father]] <= handleArray[handle]) break;
      pull(i, father);
      i=father;
    }
  heap[i]=handle;
  handleArray[handle].pos=i;
  //assert(ass());
}

template <class T, class E>
inline void HHeap<T,E>::down_botup(long i, long handle)
     // does the down-heap; insertion index of elem is again i, the procedure works
     // 'bottom-up'
{
  long j;
  long root;
  root=i;
  j=leftson(i);
  while(j<size)
    { 
      j = minimum(j);
      pull(i,j);
      i=j;
      j=leftson(i);
    }
  up(i,handle,root);
}

// public:

template <class T, class E>
HHeap<T,E>::HHeap(long m) : 
size(0), heap(m), handleArray(m), firstFreeHandle(-1)
{}

template <class T, class E>
void HHeap<T,E>::clear(void)
{
  firstFreeHandle=-1;
  size=0;
}

template <class T, class E>
long HHeap<T,E>::ins(const T& cont, E value, bool heapi)
     // inserts a new elem with value value into the heap
     // elements with value >= bound can be deleted
     // heapi = false => heap property is not preserved 
{ 
  long handle;
  if (firstFreeHandle>=0)
    {
      handle = firstFreeHandle;
      firstFreeHandle=handleArray[firstFreeHandle].pos;
    }
  else handle=size;
  handleArray[handle].cont=cont;
  handleArray[handle].value=value;
  if (!heapi) 
    {
      heap[size]=handle;
      handleArray[handle].pos=size;
      size++;
      return handle;
    }
  size++;
  up(size-1, handle);
  return handle;
}

template <class T, class E>
void HHeap<T,E>::heapify(void)
{
  long i;
  for (i=(size-2)/2; i>=0; i--)
    {
      down_botup(i,heap[i]); 
    }
  //print();
}

template <class T, class E>
long HHeap<T,E>::del(void)
     // delete the first element of the heap; 
{ 
  if (!size) return -1;
  handleArray[heap[0]].pos = firstFreeHandle;
  firstFreeHandle=heap[0];
  size--;
  if (size==0) return firstFreeHandle;
  down_botup(0,heap[size]); 
  return firstFreeHandle;
}

template <class T, class E>
long HHeap<T,E>::del(long handle)
     // delete the given element 
{ 
  if (!size) return -1;
  assert(heap[handleArray[handle].pos]==handle);
  handleArray[handle].pos = firstFreeHandle;
  firstFreeHandle=handle;
  size--;
  if (size==0) return firstFreeHandle;
  down_botup(handle,heap[size]); 
  return firstFreeHandle;
}

template <class T, class E>
void HHeap<T,E>::increase_min(E value)
     // increase the minimum
{ 
  assert(!empty());
  handleArray[heap[0]].value=value;
  down_botup(0,heap[0]); 
}


template <class T, class E>
void HHeap<T,E>::decrease_value(long handle, E value)
     // decrease cont of element handle 
{ 
  assert(!empty());
  long index=handleArray[handle].pos;
  assert((index>=0)&&(index<size));
  assert(handleArray[handle]>=value);
  handleArray[handle].value=value;
  up(index, handle);
}

/*
template <class T, class E>
void HHeap<T,E>::decrement_value(long handle)
     // decrease cont of element handle 
{ 
  assert(!empty());
  long index=handleArray[handle].pos;
  assert((index>=0)&&(index<size));
  assert(handleArray[handle]>=value);
  handleArray[handle].value--;
  up(index, handle);
}


template <class T, class E>
E HHeap<T,E>::get_value(long handle)
     // decrease cont of element handle 
{ 
  assert(!empty());
  return handleArray[handle].value;
}
*/


template <class T, class E>
inline bool HHeap<T,E>::empty(void) 
     // is the heap empty?
{ 
  return !size;
}

template <class T, class E>
inline bool HHeap<T,E>::min(E& value, T& cont)
     // returns minimal value and cont
     // true  if everything's okay, false if heap is empty
{
  if (empty()) return false;
  value = handleArray[heap[0]].value;
  cont = handleArray[heap[0]].cont;
  return true;
}

template <class T, class E>
inline T HHeap<T,E>::min(E& value)
     // returns minimal value and cont
{
  assert(!empty());
  value = handleArray[heap[0]].value;
  return handleArray[heap[0]].cont;
}

template <class T, class E>
inline T HHeap<T,E>::min(void)
     // returns cont with minimal value
{
  assert(!empty());
  return handleArray[heap[0]].cont;
}

template <class T, class E>
inline E HHeap<T,E>::minValue(T& cont)
     // returns minimal value and cont
{
  assert(!empty());
  cont = handleArray[heap[0]].cont;
  return handleArray[heap[0]].value;
}

template <class T, class E>
inline E HHeap<T,E>::minValue(void)
     // returns minimal value
{
  assert(!empty());
  return handleArray[heap[0]].value;
}

template <class T, class E>
bool HHeap<T,E>::ass(void)
{
  bool ret=true;
  for(long i=0; i<size; i++)
    {
      long j = leftson(i);
      for (long k=j; k<j+2; k++)
	{
	  if (k>=size) goto END; 
	    if (handleArray[heap[k]]<handleArray[heap[i]])
	      {
		ret = false;
		goto END;
	      }
	}
    }
END:
  if (!ret) print();
  return ret;
}

template <class T, class E>
void HHeap<T,E>::print(void)
{
  cout << "Heapsize " << size << endl;
  for (long i=0; i<size; i++)
    {
      cout << i << " handle " << heap[i] << " "; 
      handleArray[heap[i]].print();
    }
}

#endif
