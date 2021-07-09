#include "general_includes.h"


const long ENR=999; 
const int INFTY=100000000;
const double EPSILON=1./INFTY;




/**********************
TEMPLATE FUNCTIONS


template<class T>
void swap(T& a, T& b)
{
  T h=a;
  a=b;
  b=a;
}


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
  assert(r==1);
  if (t<0) t+=p;
  return t;
}


template <class T>
T gcd(T a, T b, T& x, T& y)
//extended euclidean algorithm, xa+yb=gcd(a,b)
// returns gcd(a,b)
{
  T 
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
void matrixCopy(int n, int m, T** p, T** _p)
{
  int i,j;
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      p[i][j]=_p[i][j];
}

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

************************/

/*
char* itoa(int i)
{
  int j=1;
  int l=1;
  int a=i;
  if (a<0) a *= -1;
  do
    {
      a/=10;
      l++;
    }
  while (a>0);
  a=i;
  if (a<0) 
    {
      l++;
      a *= -1;
    }
  str[--l]=0;
  while (a>9)
    {
      assert(l-j>=0);
      str[l-j]=a%10+char('0');
      a/=10;
      j++;
    }
  if (i<0)
    {
      str[1]=a%10+char('0');
      str[0]=char('-');
    }
  else str[0]=a%10+char('0');
  return str;
}
*/


void splitString(const std::string&  s, char delim, std::vector<std::string>& elems) 
// splits string s into seperate components according to the specified delimiter
{
  std::stringstream ss(s);
  std::string item;
  while(getline(ss, item, delim)) 
    elems.push_back(item);
}


void error(const char* file, int line, const char* message, long exitNumber)
{
  if (file&&line)
    std::cerr << file << " " << line << std::endl << message << std::endl << std::flush;
  else 
    std::cerr << message << std::endl << std::flush;
  exit(exitNumber);
}

double now(long long *t) 
{    
  /*
  if (t) *t= clock();
  return (double) (clock()*1./CLOCKS_PER_SEC);
  */

  tms cpuTime;
  times(&cpuTime);
  //return (double)cpuTime.tms_utime *1. / (double)CLK_TCK; // /100;
  //return ((double) cpuTime.tms_utime) *1./HZ;
  return ((double) cpuTime.tms_utime + cpuTime.tms_stime) *1./HZ;
} 

double convert(long long t) 
{
  //cout << "CLOCK TICKS -> " << CLOCKS_PER_SEC << endl;
  return (double) (t*1./100); //CLOCKS_PER_SEC);     // (float)CLK_TCK; // /100;
} 


/*
#if ! defined(false)
long round(double a)
{
  return long(floor(a+.5));
}
#endif
*/


int moreInt(double a, double b)
{
  return fabs(a-round(a)) < fabs(b-round(b));
}

int integer(double a)
{
  return int(fabs(a-round(a))<EPSILON);
}

int myRand(int mini, int maxi);


bool acceptProb(double p)
{
  static long inter=1000000;
  assert(p>=0.);
  assert(p<=1.);
  long cutPoint=round(inter*p);
  long ra=myRand(0,inter-1);
  //  std::cout << "0 <= " << ra << " <> " << cutPoint << " <= " << inter << "\n";
  if (ra<cutPoint)
    return true;
  return false;
}

long triangularIndex(long a, long b, long n)
{
  assert(a>=0);
  assert(a<n);
  assert(b>=0);
  assert(b<n);

  if (a>b)
    {
      long h=a;
      a=b;
      b=h;
    }
  assert(b>=a);
  return a*(n-1)-(a*(a-1))/2+b;
}

CRandomMersenne RGEN(0);

int myRand(int mini, int maxi)
{
  return RGEN.IRandomX(mini,maxi);
}

void mySRand(int seed)
{
  RGEN.RandomInit(seed);
}

#include "mother.cpp"
#include "mersenne.cpp"
//#include "sfmt.cpp"
