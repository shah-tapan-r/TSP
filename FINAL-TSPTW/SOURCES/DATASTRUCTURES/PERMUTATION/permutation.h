#ifndef PERMUTATION_H
#define PERMUTATION_H

#include <iostream>
#include <general_includes.h>
#include <assert.h>


//enum bool {false = 0, true = 1};

template <class T>
class Permutation
{
private:
  const T _n;
  T x;
  T * _position;
  T * _permutation;
  
  bool perm(T i);
  
  void exchange(T& i, T& j)
    {
      x=i;
      i=j;
      j=x;
    };	
  
public:
  Permutation(T n);
  ~Permutation(void);
  
  void init(void);
  
  Permutation& operator++(void);
  T operator[](T i) const;
  void permute(void)
    {
      assert(_permutation);
      assert(_n);
      for (long i=0; i<_n-1; i++)
	{
	  x=myRand(i,_n-1);
	  if (i==x) continue;
	  assert(x>i);
	  assert(x<_n);
	  exchange(_permutation[i],_permutation[x]);
	}
    };
  bool next(void);  
  //  friend ostream& operator<< (ostream& os, Permutation& p);
};



template <class T>
bool Permutation<T>::perm(T i)
{
  T j;
  bool r;
  
  if (i == _n)
    {
      //permute();
      return false;
    }
  
  if (_position[i] + 1 == _n - i)
    {
      // Folge ohne i permutieren
      r = perm(i + 1);
      // cout << "rec" << * this << endl;
      for (j = _n - i - 1; j > 0; j--) 
	{
	  _permutation[j] = _permutation[j - 1];
	  _position[_permutation[j]] = j;
	}
      _permutation[0] = i;
      _position[i] = 0;
      // cout << "rec" << * this << endl;
      return r;
    }
  
  // Zahl i um eins nach rechts schieben
  _permutation[_position[i]] = _permutation[(_position[i] + 1) % (_n - i)];
  _position[_permutation[_position[i]]] = _position[i];
  _position[i] = (_position[i] + 1) % (_n - i);
  _permutation[_position[i]] = i;
  return true;
}

template <class T>
Permutation<T>::Permutation(T n) : _n(n)
{
  _permutation = new T[_n];
  _position = new T[_n];
  init();
}

template <class T>
void Permutation<T>::init(void)
{
  T i;
  for (i = 0; i < _n; i++)
    {
      _permutation[i] = i;
      _position[i] = i;
    }
  //permute();
}

template <class T>
Permutation<T>::~Permutation(void)
{
  delete[] _position;
  delete[] _permutation;
}

template <class T>
Permutation<T>& Permutation<T>::operator++(void)
{
  if (!perm(0)) init();
  return * this;
}

template <class T>
bool Permutation<T>::next(void)
{
  return perm(0);
}

template <class T>
T Permutation<T>::operator[](T i) const
{
  return _permutation[i];
}

/*
template <class T>
ostream& operator<< (ostream& os, Permutation<T>& p)
{
  T i;
  
  os << "[ ";
  for (i = 0; i < p._n; i++)
    os << p._permutation[i] << " ";
  os << "]";
  
#ifndef NDEBUG
  os << " [ ";
  for (i = 0; i < p._n; i++)
    os << p._position[i] << " ";
  os << "]";
#endif
  
  return os;
}
*/

#endif
