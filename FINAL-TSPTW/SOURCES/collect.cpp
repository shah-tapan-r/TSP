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
#include <algorithm>
#include <math.h>
#include <assert.h>
#include <limits>
#include <sys/time.h>

#include "general_includes.h"
#include "SET/set.h"

using namespace std;

void usage(char* prog)
{
  cout << prog << " [drawfile] [seedfile]\n";
  exit (999);
}


int permhash(int* x, int n)
{
  int h=0;
  for (int i=0; i<n; i++)
    {
      int a=(x[i]*i)%13931;
      h = (h+a)%13931;
    }
  return h;
}



int main(int argc, char** argv)
{
  if (argc!=3) usage(argv[0]);

  
  ifstream inf(argv[2]);
  string temp;
  int max=-1;
  double minVal=1;
  int n;
  bool nFound=false;
  while( getline(inf,temp) )
    {
      vector<string> elems;
      splitString(temp, ',', elems);
      if (!nFound) 
	{
	  n=elems.size()-2;
	  if (n<=0) exit(1);
	  nFound=true;
	}
      if (elems.size()!=n+2) 
	{
	  //std::cout << "too short\n";
	  continue;
	}
      int cur = atoi(elems[0].c_str());
      if (max<0) n=elems.size()-2;
      double newVal=atof(elems[n+1].c_str());
      assert(max<0||newVal<=minVal);
      max = maximum(max,cur);
      minVal=newVal;
    }
  inf.close();

  if (max<0) cout << "No Seedfile Found\n";
  else cout << "Current seed max -> " << max << " has value " << minVal << endl;

  SSet recorded(13931);
  if (max>0)
    {
      int x[n];
      ifstream inf2(argv[2]);
      while( getline(inf2,temp) )
	{
	  vector<string> elems;
	  splitString(temp, ',', elems);
	  if (elems.size()!=n+2) 
	    {
	      //std::cout << "too short\n";
	      continue;
	    }
	  int cur = atoi(elems[0].c_str());
	  if (max<0) n=elems.size()-2;
	  double newVal=atof(elems[n+1].c_str());
	  if (newVal>minVal+EPSILON) continue;
	  for (int i=0; i<n; i++)
	    x[i]=atoi(elems[i+1].c_str());
	  recorded.add(permhash(x,n));
	  //      std::cout << "recording with hash " << permhash(x,n) << std::endl;
	}
      inf2.close();
    }
  
  ifstream logf(argv[1]);
  bool succ=false;
  //std::cout << "trying infile\n ";
  int lineNum=0;
  int* thesis=0;
  int* y=0;
  while( getline(logf,temp) )
    {
      lineNum++;
      //std::cout << "line " << lineNum << ": ";
      
      vector<string> elems;
      splitString(temp, ',', elems);
      
      if (!thesis) 
	{
	  if (max<0) n=elems.size()-2; 
	  thesis=new int[n];
	  y=new int[n];
	}
      if (elems.size()!=n+2) 
	{
	  //std::cout << "too short\n";
	  continue;
	}
      double newVal = atof(elems[n+1].c_str());
      if ((max>=0) && (newVal>minVal+EPSILON) )
	{
	  //std::cout << "old vector\n"; 
	      continue;
	}
      //std::cout << "success\n";
      for (int i=0; i<n; i++)
	y[i]=atoi(elems[i+1].c_str());
      //      std::cout << "New hash -> " << permhash(x,n) << std::endl;
      if (!recorded.isIn(permhash(y,n)))
	{
	  for (int i=0; i<n; i++)
	    thesis[i]=y[i];
	  succ=true;
	  if (minVal>newVal) minVal=newVal;
	}
      //else std::cout << "Already exists - No success\n";
    }
  logf.close();

 
  if (!succ) 
    {
      cout << "No improving seed found in drawfile\n";
      delete [] thesis;
      delete [] y;
      return 0;
    }
  cout << "Best seed in drawfile has value " << minVal << endl;
  
  ofstream seedf(argv[2],ios::out | ios::app);
  seedf << ++max << "," << thesis[0];
  for (int i=1; i<n; i++)
    seedf << "," << thesis[i];
  seedf << "," << minVal << std::endl;	  
  seedf.close();

  delete [] thesis;
  return 0;
}
