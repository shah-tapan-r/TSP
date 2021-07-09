
#ifndef SET_H
#define SET_H

//#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

class SSet
{

  void alloc(void)
  {
    marked = new long[maxCard];
    list = new long[maxCard];
  }
  
  void freeMem(void)
  {
    delete [] marked; marked=0;
    delete [] list; list=0;
  }

  void init(void)
  {
    long j;
    for (j=0; j<maxCard; j++)
      marked[j]=-1;
  }


public: 
  long* marked;
  
  static long StandardMaxCard;

  long* list;
  long card;
  long maxCard;

  //SSet(void) : maxCard(0), card(0), list(0), marked(0) {}

 SSet(long mc=StandardMaxCard, bool all=false) : marked(0), list(0), card(0), maxCard(mc)
  {
    if (maxCard==0) return;
    alloc();
    if (all) initAll();
    else init();
  }

  ~SSet(void)
  {
    freeMem();
  }

  void setDimension(long mc, bool all=false)
  {
    assert(maxCard==0);
    maxCard=mc;
    alloc();
    if (all) initAll();
    else init();
  }

 SSet(const SSet& s) : card(s.card), maxCard(s.maxCard)
    {
      if (this==&s) return;
      alloc();
      for (long j=0; j<maxCard; j++)
	{
	  marked[j]=s.marked[j];
	  list[j]=s.list[j];
	}
    }

  long operator[](const long& i) const
  {
    assert(i>=0);
    assert(i<card);
    return list[i];
  }

  SSet& operator=(const SSet& s)
    {
      
      if (s.maxCard>maxCard) 
	{
	  freeMem();
	  maxCard=s.maxCard;
	  alloc();
	}
      else maxCard=s.maxCard;
      for (long j=0; j<maxCard; j++)
	{
	  marked[j]=s.marked[j];
	  list[j]=s.list[j];
	}
      card=s.card;
      /*
      long j;
      for (j=0; j<card; j++)
	marked[list[j]]=-1;
      card=s.card;
      for (j=0; j<s.card; j++)
	{
	  list[j]=s.list[j];
	  marked[list[j]]=j;
	}
      */
      return *this;
    }

  void initAll(void)
  {
    long j;
    for (j=0; j<maxCard; j++)
      {
	list[j]=j;
	marked[j]=j;
      }
    card=maxCard;
  }

  bool add(long item)
    // adds an item to the set, returns false if item was in set before
  {
    assert(item>=0);
    assert(item<maxCard);
    if(marked[item]!=-1) return false;
    marked[item]=card;
    list[card++]=item;
    return true;
  }

  bool rem(long item)
    // removes an item, returns false if item was not in the set
  {
    assert(item>=0);
    assert(item<maxCard);
    if(marked[item]==-1) return false;
    list[marked[item]]=list[--card];
    marked[list[marked[item]]]=marked[item];
    marked[item]=-1;
    return true;
  }

  void rem(long* disAllowed, long k, long* removed, long& numberOfRemoved)
  {
    long i=0;
    long item;
    numberOfRemoved=0;
    for (i=0; i<k; i++)
    {
      item=disAllowed[i];
      if (!isInSet(item)) continue;
      removed[numberOfRemoved++]=item;
      rem(item);
    }
  }

  void remAll(void)
  {
    long j;
    for (j=0; j<card; j++)
      marked[list[j]]=-1;
    card=0;
  }

  bool isInSet(long item) const
  {
    if(marked[item]==-1) return false;
    return true;
  }

  bool isIn(long item) const
  {
    return isInSet(item);
  }

  SSet& operator+=(const SSet& s)  // set union
  {
    assert(maxCard==s.maxCard);
    long i,item;
    for (i=0; i<s.card; i++)
      {
	item=s.list[i];
	if(marked[item]==-1)
	  {
	    marked[item]=card;
	    list[card++]=item;
	  }
      }
    return *this;
  }

  SSet& operator-=(const SSet& s)  // remove s from this set
  {
    assert(maxCard==s.maxCard);
    long i;
    for (i=0; i<s.card; i++)
      {
	rem(s.list[i]);
	/*
	long item=s.list[i];
	assert(item>=0);
	assert(item<maxCard);
	if(marked[item]==-1) continue;
	list[marked[item]]=list[--card];
	marked[list[marked[item]]]=marked[item];
	marked[item]=-1;
	*/
      }
    return *this;
  }

  bool intersectsWith(const SSet& s) 
  {
    assert(maxCard==s.maxCard);
    long i;
    if (s.card<card)
      {
	for (i=0; i<s.card; i++)
	  if (isInSet(s.list[i]))
	    return true;
	return false;
      }
    for (i=0; i<card; i++)
      if (s.isInSet(list[i]))
	return true;
    return false;

  }

  SSet& operator*=(const SSet& s)  // set intersection
  {
    assert(maxCard==s.maxCard);
    long i,item;
    i=0;
    while (i<card)
      {
	item=list[i];
	assert(item>=0);
	assert(item<maxCard);
	if (s.isInSet(item)) 
	  {
	    i++;
	    continue;
	  }
	assert(marked[item]==i);
	list[i]=list[--card];
	marked[list[i]]=i;
	marked[item]=-1;
      }
    return *this;
  }

  SSet& operator*=(bool* b)  // set intersection
    // intersects with given set and empties the input set completely at the same time
  {
    long i,item;
    i=0;
    while (i<card)
      {
	item=list[i];
	assert(item>=0);
	assert(item<maxCard);
	if (b[item]) 
	  {
	    i++;
            b[item]=false;
	    continue;
	  }
	assert(marked[item]==i);
	list[i]=list[--card];
	marked[list[i]]=i;
	marked[item]=-1;
      }
    return *this;
  }

  void removeAllBut(int item, int* removed, int& numberOfRemoved)
  {
    assert(marked[item]!=-1);
    list[marked[item]]=list[--card];
    marked[list[marked[item]]]=marked[item];
    marked[item]=-1;
    numberOfRemoved=0; 
    for (long i=0; i<card; i++)
    {
      removed[numberOfRemoved++]=list[i];
      marked[list[i]]=-1;
    }
    card=0;
    marked[item]=card;
    list[card++]=item;     
  }

  void removeAllBut(bool* allowed, int* removed, int& numberOfRemoved)
  {
    long i=0;
    long item;
    numberOfRemoved=0;
    while(i<card)
    {
      if (allowed[list[i]]) 
      {
	i++;
	continue; 
      }
      removed[numberOfRemoved++]=list[i];

      item=list[i];
      list[i]=list[--card];
      marked[list[i]]=i;
      marked[item]=-1;
    }
  }

  friend std::ostream& operator<<(std::ostream& out, const SSet& s);
  friend bool operator==(const SSet& s, const SSet& t);
  friend bool operator!=(const SSet& s, const SSet& t);
  friend bool operator<=(const SSet& s, const SSet& t);
  friend bool operator>=(const SSet& s, const SSet& t);
  friend bool operator<(const SSet& s, const SSet& t);
  friend bool operator>(const SSet& s, const SSet& t);

};


#endif

