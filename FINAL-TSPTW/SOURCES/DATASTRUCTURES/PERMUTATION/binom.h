#ifndef BINOM_H
#define BINOM_H

#include "general_includes.h"

template <class T>
class Binom
{
  T n,k;
  T *subset;
  T *fullset; 

public:
 Binom(T _n, T _k) : n(_n), k(_k), subset(0), fullset(0)  // n choose k
    {
      assert(n>=k);
      if (k>0) subset = new T[k];
      else subset = 0;
      fullset = new T[n];
      init();
    }
  ~Binom(void)
    {
      if (subset) delete [] subset;
      if (fullset) delete [] fullset;
    }
  inline T& operator[](T i)
    {
      assert((i>=0)&&(i<k));
      return subset[i];
    }
  inline bool next(void)
    {
      T i = k-1;
      while ((i>=0)&&(subset[i]==n-k+i)) i--;
      if (i<0) 
	{
	  init();
	  return false;
	}
      assert(subset[i]<n-k+i);
      assert((i==k-1)||(subset[i]<subset[i+1]-1));
      subset[i]++;
      for (++i; i<k; i++)
	subset[i]=subset[i-1]+1;
      return true;
    }
  inline void init(void)
    {
      for (T i=0; i<k; i++)
	subset[i]=i;
    }
  inline void exchange(T& i, T& j)
    {
      static T x;
      x=i;
      i=j;
      j=x;
    };
  inline void select(bool sort=true)
    {
      T i,h;
      for (i=0; i<n; i++)
	fullset[i]=i;
      for (i=0; i<k; i++)
	{
	  exchange(fullset[i], fullset[myRand(i,n-1)]);
	  subset[i]=fullset[i];
	}
      if (!sort) return;
      // now sort the result
      for (i=0; i<n; i++)
	fullset[i]=0;
      for (i=0; i<k; i++)
	fullset[subset[i]]=1;
      h=0;
      for (i=0; i<n; i++)
	if (fullset[i]) subset[h++]=i;
    }
  void print(void)
  {
    for (int i=0; i<k; i++)
      std::cout << subset[i] << " ";
  }
};


#endif
