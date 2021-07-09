#ifndef GENERAL_INCLUDES_H
#define GENERAL_INCLUDES_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <sstream>

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#include <sys/param.h>
#include <sys/times.h>
#include <sys/types.h>
#include <math.h>
//#include <iostream>
//#include "sfmt.h"
#include "randomc.h"


/*
#define bool unsigned char
#if ! defined(false)
  #define false 0
  #define true !false  
#endif
*/


#define PROGERROR(x) error(__FILE__, __LINE__, x, ENR)
#define RUNTIMEERROR(x) error(__FILE__, __LINE__, x, ENR)
#define RUNERROR(x) error(__FILE__, __LINE__, x, ENR)

typedef int (*CompareFct) (const void *, const void *);

extern const long ENR; 
extern const int INFTY;
extern const double EPSILON;


extern int myRand(int mini, int maxi);
extern void mySRand(int seed);
extern void error(const char* file, int line, const char* message, long exitNumber);
extern double now(long long* t=0);
extern double convert(long long t);

/*
#if ! defined(false)
long round(double a);
#endif
*/


void splitString(const std::string&  s, char delim, std::vector<std::string>& elems);


template <class T>
T sqr(T a)
{
  return a*a;
}

template <class T>
int chooseDist(T* dis, int n)
{
  if (n<=0) return -1;
  long long* D = new long long[n];
  T acc=0.;
  for (int i=0; i<n; i++)
    {
      acc+=dis[i];
      D[i]=(int) minimum(acc*1000000.,1000000.);
    }
  D[n-1]=1000000;
  long long r=rand()%1000000;
  int x;
  for (x=0;x<n; x++)
    if (r<D[x]) break;  
  assert(x<n);
  delete [] D;
  return x;
}

int moreInt(double a, double b);
int integer(double a);

template<class T>
T inverse(T a, T p)
{
  T t=0; T nt=1;
  T r=p; T nr=a;
  while (nr!=0)
    {
      T q=r/nr;
      T h=t;
      t=nt;
      nt=h-q*nt;
      h=r;
      r=nr;
      nr=h-q*nr;
    }
  if (r!=1)
    {
      std::cout << a << " " << p << std::endl;
    }
  assert(r==1);
  if (t<0) t+=p;
  return t;
}


template<class T, class S>
T maximum(const T& a, const S& b)
{
  if (a<b) return b;
  return a;
}

template<class T, class S>
T minimum(const T& a, const S& b)
{
  if (a>b) return b;
  return a;
}

template <class T>  
void copy(long l, T* c, T* o)
{
  for (long i=0;i<l;i++)
    c[i]=o[i];  
}

template <class T>
void matrixCopy(int n, int m, T** p, T** _p);

template <class T>
void matrixAlloc(int n, int m, T**& matrix)
{
  int i;
  matrix=new T*[n];
  for (i=0; i<n; i++)
    {
      matrix[i]=new T[m];
    }
}

template <class T>
void matrixFree(int n, T**& matrix)
{
  int i;
  for (i=0; i<n; i++)
    delete [] matrix[i];
  delete [] matrix;
  matrix = 0;
}

bool acceptProb(double p);

long triangularIndex(long a, long b, long n);


#endif
