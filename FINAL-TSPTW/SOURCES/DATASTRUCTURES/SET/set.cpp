

//#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "set.h"


bool operator==(const SSet& s, const SSet& t)
{
  if (s.maxCard!=t.maxCard) return false;
  if (s.card!=t.card) return false;
  for (long j=0; j<t.maxCard; j++)
    if (t.isIn(j)!=s.isIn(j)) return false;
  return true;
}

bool operator!=(const SSet& s, const SSet& t)
{
  return !(s==t);
}

bool operator<=(const SSet& s, const SSet& t)
{
  if (s.maxCard!=t.maxCard) return false;
  if (s.card>t.card) return false;
  for (long j=0; j<s.card; j++)
    if (!t.isIn(s.list[j])) return false;
  return true;
}

int compare(const SSet& s, const SSet& t)
// -1 s<t
//  0 s==t
//  1 s>t 
//  2 incomparable
{
  if (s.maxCard!=t.maxCard) return 2;
  if (s.card>t.card) 
    {
      for (long j=0; j<t.card; j++)
	if (!s.isIn(t.list[j])) return 2;
      return 1;
    }
  if (s.card<t.card)
    {
      for (long j=0; j<s.card; j++)
	if (!t.isIn(s.list[j])) return 2;
      return -1;
    }
  assert(s.card==t.card);
  for (long j=0; j<s.card; j++)
    if (!t.isIn(s.list[j])) return 2;
  return 0;
}

bool operator>=(const SSet& s, const SSet& t)
{
  return t<=s;
}

bool operator<(const SSet& s, const SSet& t)
{
  if (s.maxCard!=t.maxCard) return false;
  if (s.card>=t.card) return false;
  for (long j=0; j<s.card; j++)
    if (!t.isIn(s.list[j])) return false;
  assert(!(s==t));
  assert(s<=t);
  return true;
}

bool operator>(const SSet& s, const SSet& t)
{
  return t<s;
}


std::ostream& operator<<(std::ostream& out, const SSet& s)
{
  for (long i=0; i<s.card; i++)
    out << s.list[i] << " ";
  return out;
}


long SSet::StandardMaxCard=0;


