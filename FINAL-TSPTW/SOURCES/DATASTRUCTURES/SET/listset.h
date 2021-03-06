
#ifndef LISTSET_H
#define LISTSET_H

//#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <string.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

#include "listhack.h"


class ListSet
{

  void alloc(void)
  {
//     marked = new long[maxCard];
//     list = new long[maxCard];
  }
  
  void freeMem(void)
  {
//     delete [] marked; marked=0;
//     delete [] list; list=0;
  }

  void init(void)
  {
//     long j;
//     for (j=0; j<maxCard; j++)
//       marked[j]=-1;
  }


public: 

//   std::unordered_map<long,long> hash;
//   std::set<long> data;
  std::vector<long> data;

  long* marked;
  
  static long StandardMaxCard;

  ListHack list;
  long card;

  //SSet(void) : maxCard(0), card(0), list(0), marked(0) {}

 ListSet(long mc=StandardMaxCard, bool all=false) : data(std::vector<long>()), marked(0), list(this), card(0)
  {
//     if (maxCard==0) return;
//     alloc();
//     if (all) initAll();
//     else init();
  }

  ~ListSet(void)
  {
//     if (maxCard==0) return;
//     freeMem();
  }

  void setDimension(long mc, bool all=false)
  {
//     assert(maxCard==0);
//     maxCard=mc;
//     alloc();
//     if (all) initAll();
//     else init();
  }

 ListSet(const ListSet& s) : data(s.data), list(ListHack(this)), card(0)
//      : card(s.card), maxCard(s.maxCard)
    {
        std::cout << "BAD" << std::endl;
      if (this==&s) return;
      alloc();
//       for (long j=0; j<maxCard; j++)
// 	{
// 	  marked[j]=s.marked[j];
// 	  list[j]=s.list[j];
// 	}
    }

  long operator[](const long& i) //const
  {
      return list[i];
//     assert(i>=0);
//     assert(i<card);
//     return list[i];
  }

  ListSet& operator=(const ListSet& s)
    {
      
        std::cout << "BAD" << std::endl;
//       if (s.maxCard>maxCard) 
// 	{
// 	  freeMem();
// 	  maxCard=s.maxCard;
// 	  alloc();
// 	}
//       else maxCard=s.maxCard;
//       for (long j=0; j<maxCard; j++)
// 	{
// 	  marked[j]=s.marked[j];
// 	  list[j]=s.list[j];
// 	}
//       card=s.card;
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
        std::cout << "BAD" << std::endl;
//     long j;
//     for (j=0; j<maxCard; j++)
//       {
// 	list[j]=j;
// 	marked[j]=j;
//       }
//     card=maxCard;
  }

  bool add(long item)
    // adds an item to the set, returns false if item was in set before
  {
      auto pos = std::find(data.begin(), data.end(), item);
      if(pos == data.end()) {
          data.push_back(item);
//       if(data.count(item) > 0) return false;
//       data.insert(item);
          card++;
//           std::sort(data.begin(), data.end()); // haha could I make this any slower? :)
          return true;
      } else {
          return false;
      }
  }

  bool rem(long item)
    // removes an item, returns false if item was not in the set
  {
      auto pos = std::find(data.begin(), data.end(), item);
      if(pos != data.end()) {
          data.erase(pos);
          card--;
          return true;
      }
      return false;
//       if(data.count(item) == 0) return false;
//       data.erase(item);
//       card--;
//       return true;
  }

//   void rem(long* disAllowed, long k, long* removed, long& numberOfRemoved)
//   {
//     long i=0;
//     long item;
//     numberOfRemoved=0;
//     for (i=0; i<k; i++)
//     {
//       item=disAllowed[i];
//       if (!isInSet(item)) continue;
//       removed[numberOfRemoved++]=item;
//       rem(item);
//     }
//   }

  void remAll(void)
  {
      data.clear();
      card = 0;
  }

  bool isInSet(long item) const
  {
      auto pos = std::find(data.begin(), data.end(), item);
//       return data.count(item) > 0;
      return pos != data.end();
  }

  bool isIn(long item) const
  {
    return isInSet(item);
  }

  ListSet& operator+=(const ListSet& s)  // set union
  {
    for(auto ii = s.data.begin(); ii!=s.data.end(); ++ii) {
          if(!isIn(*ii)) {
              data.push_back(*ii);
              card++;
          }
      }
      return *this;
//     assert(maxCard==s.maxCard);
//     long i,item;
//     for (i=0; i<s.card; i++)
//       {
// 	item=s.list[i];
// 	if(marked[item]==-1)
// 	  {
// 	    marked[item]=card;
// 	    list[card++]=item;
// 	  }
//       }
//     return *this;
  }

  ListSet& operator-=(const ListSet& s)  // remove s from this set
  {
      std::cout << "warning :( -=" << std::endl;
      return *this;
//     assert(maxCard==s.maxCard);
//     long i;
//     for (i=0; i<s.card; i++)
//       {
// 	rem(s.list[i]);
// 	/*
// 	long item=s.list[i];
// 	assert(item>=0);
// 	assert(item<maxCard);
// 	if(marked[item]==-1) continue;
// 	list[marked[item]]=list[--card];
// 	marked[list[marked[item]]]=marked[item];
// 	marked[item]=-1;
// 	*/
//       }
//     return *this;
  }

  bool intersectsWith(const ListSet& s) 
  {
//     assert(maxCard==s.maxCard);
//     long i;
//     if (s.card<card)
//       {
// 	for (i=0; i<s.card; i++)
// 	  if (isInSet(s.list[i]))
// 	    return true;
// 	return false;
//       }
//     for (i=0; i<card; i++)
//       if (s.isInSet(list[i]))
// 	return true;
    return false;

  }

  ListSet& operator*=(const ListSet& s)  // set intersection
  {
      std::cout << "warning :( *=" << std::endl;
//     assert(maxCard==s.maxCard);
//     long i,item;
//     i=0;
//     while (i<card)
//       {
// 	item=list[i];
// 	assert(item>=0);
// 	assert(item<maxCard);
// 	if (s.isInSet(item)) 
// 	  {
// 	    i++;
// 	    continue;
// 	  }
// 	assert(marked[item]==i);
// 	list[i]=list[--card];
// 	marked[list[i]]=i;
// 	marked[item]=-1;
//       }
    return *this;
  }

  ListSet& operator*=(bool* b)  // set intersection
    // intersects with given set and empties the input set completely at the same time
  {
      std::cout << "warning :( *= bool" << std::endl;
//     long i,item;
//     i=0;
//     while (i<card)
//       {
// 	item=list[i];
// 	assert(item>=0);
// 	assert(item<maxCard);
// 	if (b[item]) 
// 	  {
// 	    i++;
//             b[item]=false;
// 	    continue;
// 	  }
// 	assert(marked[item]==i);
// 	list[i]=list[--card];
// 	marked[list[i]]=i;
// 	marked[item]=-1;
//       }
    return *this;
  }

  void removeAllBut(int item, int* removed, int& numberOfRemoved)
  {
//     assert(marked[item]!=-1);
//     list[marked[item]]=list[--card];
//     marked[list[marked[item]]]=marked[item];
//     marked[item]=-1;
//     numberOfRemoved=0; 
//     for (long i=0; i<card; i++)
//     {
//       removed[numberOfRemoved++]=list[i];
//       marked[list[i]]=-1;
//     }
//     card=0;
//     marked[item]=card;
//     list[card++]=item;     
  }

  void removeAllBut(bool* allowed, int* removed, int& numberOfRemoved)
  {
//     long i=0;
//     long item;
//     numberOfRemoved=0;
//     while(i<card)
//     {
//       if (allowed[list[i]]) 
//       {
// 	i++;
// 	continue; 
//       }
//       removed[numberOfRemoved++]=list[i];
// 
//       item=list[i];
//       list[i]=list[--card];
//       marked[list[i]]=i;
//       marked[item]=-1;
//     }
  }

};



#endif

