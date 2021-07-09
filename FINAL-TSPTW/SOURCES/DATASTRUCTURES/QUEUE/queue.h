#ifndef QUEUE_H
#define QUEUE_H


#include <general_includes.h>
#include <ARRAY/array.h>
#include <assert.h>
#include <iostream>

template <class T> class Queue;

template <class T> std::ostream& operator<<(std::ostream& out, Queue<T>& q);


template <class T>
class QueueIterator
{
  long ind;
  Queue<T>* queue;
 public:
 QueueIterator(Queue<T>& q) : queue(&q), ind(q.first) {}
  bool ok(void) { return ind>=0; }
  QueueIterator& operator++(void)
  {
    if (!ok()) return *this;
    ind=queue->next[ind];
    return *this;
  }
  T& operator*()
  {
    return queue->queue[ind];
  }

  friend class Queue<T>;
};

template <class T>
class Queue
{
  Array<T> queue;
  Array<long> next;
  Array<long> deleted;
  long first, last;
  long size;
  long vacant; 

public:

 Queue(long maxSize=1000) : queue(maxSize), next(maxSize), deleted(maxSize)
    {
      reset();
    }
  ~Queue(void)
    {}
  void enq(const T& element)
    {
      //std::cout << "Enq -> " << *this << std::endl;
      if (vacant)
	{
	  assert(vacant>0);
	  assert(deleted[vacant-1]>=0);
	  assert(deleted[vacant-1]<queue.getSize());
	  long ind=deleted[--vacant];
	  queue[ind]=element;
	  if (last>=0) next[last]=ind;
	  next[ind]=-1;
	  if (size==0) 
	    {
	      assert(first==-1);
	      assert(last==-1);
	      first=ind;
	    }
	  last=ind;
	  size++;
	  //std::cout << "Done -> " << *this << std::endl;
	  return;
	}
      
      assert(vacant==0);
      queue[size]=element;
      if (last>=0) next[last]=size;
      next[size]=-1;
      if (size==0) 
	{
	  assert(first==-1);
	  assert(last==-1);
	  first=size;
	}
      last=size;
      size++;
      //std::cout << "Done -> " << *this << std::endl;
    }
  bool deq(T& element)
    {
      if (!size) 
	{
	  assert(first==-1);
	  assert(last==-1);
	  return false;
	}
      //std::cout << "Deq -> " << *this << std::endl;
      assert(first>=0);
      assert(last>=0);
      element=queue[first];
      deleted[vacant++]=first;
      first=next[first];
      if (--size==0) 
	{
	  assert(first==-1);
	  //last=-1;
	  reset();
	}
      //std::cout << "Done deq -> " << *this << std::endl;
      return true;
    }
  bool deq(void)
    {
      if (!size) 
	{
	  assert(first==-1);
	  assert(last==-1);
	  return false;
	}
      //std::cout << "Deq -> " << *this << std::endl;
      assert(first>=0);
      assert(last>=0);
      deleted[vacant++]=first;
      first=next[first];
      if (--size==0) 
	{
	  assert(first==-1);
	  //last=-1;
	  reset();
	}
      //std::cout << "Done deq -> " << *this << std::endl;
      return true;
    }
  T front(void)
  {
    assert(size);
    return queue[first];
  }
  void reset(void)
    {
      size=vacant=0;
      first=last=-1;
    }
  bool empty(void)
    {
      return !size;
    }
  long getSize(void)
  {
    return size;
  }
  friend std::ostream& operator<< <>(std::ostream& out, Queue<T>&);
  friend class QueueIterator<T>;
};


template <class T>
std::ostream& operator<<(std::ostream& out, Queue<T>& q)
{
  long next=q.first;
  while (next!=-1)
    {
      out << q.queue[next] << " ";
      next=q.next[next];
    }
  out << std::endl;
  return out;
}

#endif



