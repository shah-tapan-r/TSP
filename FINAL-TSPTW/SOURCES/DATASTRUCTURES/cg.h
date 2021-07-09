
#ifndef CGMATRIX_H
#define CGMATRIX_H

#include <sys/times.h>
#include <stdlib.h>
#include <iostream.h>
#include <math.h>
#include <assert.h>
#include <ilcplex/ilocplex.h>

#include "general_includes.h"

class ColGenMatrix

// min cx 
// s.t.
// Mx <=> b
//  x >= 0

{
  double **M;                       // matrix
  double *b;                        // right hand side
  long *constrType;                 // -1 = leq, 0 = eq, 1 = geq
  double *c;                        // cost per column
  long maxNoCols, noCols, noConstr; // max number of columns and matrix dimensions

                                    // cplex related variables
  IloEnv env;
  IloModel cgModel;
  IloCplex cgSolver;
  IloNumVarArray var;
  IloObjective cost;
  IloRangeArray constraint;

  void resetModel(void)
  {
    env.end();
    env=IloEnv();
    
    cgModel = IloModel(env);
    cgSolver = IloCplex(cgModel);
    cgSolver.setRootAlgorithm(IloCplex::Primal);
    cgSolver.setParam(IloCplex::SimDisplay,0);
    var = IloNumVarArray(env);
    cost = IloAdd(cgModel, IloMinimize(env));
    if (noConstr)
      {
	setRightHandSide();
	if (noCols)
	  {
	    long i,j;
	    IloNumArray column(env,noConstr);
	    for (j=0; j<noCols; j++)
	      {
		for (i=0; i<noConstr; i++)
		  column[i]=M[j][i];
		var.add(IloNumVar(cost(c[j])+constraint(column)));
	      }
	  }
      }    
  }
   
  void setRightHandSide(void)
  {
    long i;
    IloNumArray lb(env,noConstr);
    IloNumArray ub(env,noConstr);
    for (i=0; i<noConstr; i++)
      {
	if (constrType[i]<0)
	  {
	    lb[i]=-IloInfinity;
	    ub[i]=b[i];
	  }
	else if (constrType[i]>0)
	  {
	    lb[i]=b[i];
	    ub[i]=IloInfinity;
	  }
	else
	  {
	    lb[i]=b[i];
	    ub[i]=b[i];
	  }
      }      
    constraint = IloRangeArray(env,lb,ub);
    cgModel.add(constraint);    
  }
    
 public: 
  
  ColGenMatrix(long _maxNoCols=10000, long _noConstr=0, 
	       long *_constrType=(long*)0, 
	       double *_b=(double*)0) : 
    maxNoCols(_maxNoCols), noCols(0), b(_b), constrType(_constrType), 
    noConstr(_noConstr), env(), cgModel(env), var(env), cgSolver(cgModel)
  {
    M = new double*[maxNoCols];
    c = new double[maxNoCols];
    long i;
    for (i=0; i<maxNoCols; i++)
      {
	M[i]=(double*)0;
	c[i]=0.;
      }
    resetModel();
  }
    
    
  ~ColGenMatrix(void)
  {
    for (long i=0; i<maxNoCols; i++)
      delete [] M[i];
    delete [] M;
    delete [] b;
    delete [] c;
    delete [] constrType;
    env.end();
  }
  
  void addColumn(double*& col, double _cost=0.)
  {
    if (noCols>=maxNoCols)
      {
	cout << "maxcol exceeded!\n";
	double** H = new double*[2*maxNoCols];
	double* h = new double[2*maxNoCols];
	long i;
	for (i=0; i<maxNoCols; i++)
	  {
	    H[i]=M[i];
	    h[i]=c[i];
	  }
	for (; i<2*maxNoCols; i++)
	  {
	    H[i]=(double*)0;
	    h[i]=0.;
	  }
	maxNoCols *=2;  
	delete [] M;
	delete [] c;
	M=H;
	c=h;
      }
    M[noCols]=col;
    c[noCols++]=_cost;
    
    long i;
    IloNumArray column(env,noConstr);
    for (i=0; i<noConstr; i++)
      column[i]=col[i];
    var.add(IloNumVar(cost(_cost)+constraint(column)));
  } 
   
  void setUpperBounds(double* ub)
  {
    static long i;
    for (i=0; i<noCols; i++) var[i].setUb(ub[i]);
  } 
  
  void setUpperBound(long index, double ub)
  {
    assert((index>=0)&&(index<noCols));
    var[index].setUb(ub);
  }
  
  void setLowerBounds(double* lb)
  {
    static long i;
    for (i=0; i<noCols; i++) var[i].setLb(lb[i]);
  }
   
  void setLowerBound(long index, double lb)
  {
    assert((index>=0)&&(index<noCols));
    var[index].setLb(lb);
  }
  
  void setRightHandSide(double* _b, long *_constrType, long _noConstr)
  {
    assert(!b);
    //delete [] b;
    //delete [] constrType;
    b=_b;
    constrType=_constrType;
    noConstr=_noConstr;
    setRightHandSide();
  }
  
  void modifyRHSEntry(long row, double newval) 
  {
    b[row] = newval;
    constraint[row].setUb(newval);
  }
  
  void setCosts(double* _c)
  {
    delete [] c;
    c=_c;
    IloNumArray _cost(env,noCols);
    long i;
    for (i=0; i<noCols; i++)
      _cost[i]=c[i];
    cost.setCoef(var,_cost);
  }
  
  long getNumberOfColumns(void) const
  { 
    return noCols;
  } 
  
  long getMaxNumberOfColumns(void) const
  { 
    return maxNoCols;
  } 
  
  void compress(double* x, long offset=0)
  // first offset columns are protected from
  // being deleted. 
  // x[j]=0 => delete column j
  {
    long i,j=offset;
    for (i=offset; i<noCols; i++)
      {
	if (x[i]!=0)
	  {
	    if (j<i)
	      {
		assert(!M[j]);
		M[j]=M[i];
		c[j]=c[i];
		M[i] = (double*)0;
		c[i] = 0.;
	      }
	    else assert(j==i);
	    j++;
	  }
	else 
	  {
	    delete [] M[i];
	    M[i] = (double*)0;
	    c[i] = 0.;
	  }
      } 
    noCols=j;
    resetModel();
  }
  
  void compress(double* duals, double redCostLimit, long offset=0)
  // first offset columns are protected from
  // being deleted. 
  // c[j]-dualsT*M[j]>redCostLimit => delete column j
  {
    long i,j=offset;
    for (i=offset; i<noCols; i++)
      {
	if (redCosts(duals,i)<=redCostLimit)
	  {
	    if (j<i)
	      {
		assert(!M[j]);
		M[j]=M[i];
		c[j]=c[i];
		M[i] = (double*)0;
		c[i] = 0.;
	      }
	    else assert(j==i);
	    j++;
	  }
	else 
	  {
	    delete [] M[i];
	    M[i] = (double*)0;
	    c[i] = 0.;
	  }
      } 
    noCols=j;
    resetModel();
    solve();
  }
  
  double redCosts(double* duals, long j)
  // reduced costs of column j: c[j]-dualsT*M[j]
  {
    long i;
    double rc=c[j];
    for (i=0; i<noConstr; i++)
      rc-=duals[i]*M[j][i];
    return rc;
  }
  
  long readyToSolve(void) const
  {
    return b&&noCols&&noConstr&&constrType;
  }
  
  double solveOpt(double *duals=(double *)0, double *solution= (double *)0)
  {
    double result;
    result=solve(duals, solution);
    return result;
  }
  
  double solve(double *duals=(double *)0, double *solution= (double *)0)
  {
    assert(readyToSolve());
    cgSolver.solve();
    
    if(solution)
      {
	long i;
	for (i=0; i<noCols; i++)
	  solution[i]=cgSolver.getValue(var[i]);
      }
    
    if(duals)
      {
	long i;
	for (i=0; i<noConstr; i++)
	  duals[i]=cgSolver.getDual(constraint[i]);
      }
    return cgSolver.getValue(cost);
  }
  
  double solveDual(double *duals=(double *)0, double *solution= (double *)0)
  {
    assert(readyToSolve());    
    cgSolver.setRootAlgorithm(IloCplex::Dual);
    cgSolver.solve();
    
    if(solution)
      {
	long i;
	for (i=0; i<noCols; i++)
	  solution[i]=cgSolver.getValue(var[i]);
      }
    
    if(duals)
      {
	long i;
	for (i=0; i<noConstr; i++)
	  duals[i]=cgSolver.getDual(constraint[i]);
      }
    
    cgSolver.setRootAlgorithm(IloCplex::Primal);
    return cgSolver.getValue(cost);
  }
  
  void getSolution(double* solution) 
  {
    assert(solution);
    IloNumArray sol(env);
    cgSolver.getValues(sol, var);
    int i;
    for (i = 0; i < noCols; i++)
      solution[i] = sol[i];
  }
  
  void getSlacks(double *slacks) 
  {
    IloNumArray iloslacks(env, noConstr); 
    cgSolver.getSlacks(iloslacks, constraint);
    for (int i=0; i < noConstr; i++) slacks[i] = iloslacks[i];
  }
  
  void printMatrix(void) const
  {
    for (long j=0; j<noCols; j++)
      printColumn(j);
  }
  
  void printColumn(long j) const
  {
    long i;
    for (i=0; i<noConstr; i++)
      if (M[j][i]) cout << i << "," << M[j][i] << " ";
    cout << endl;
  }
  
  /*
    void printMx(double* x) const
    {
    double *s = new double[noConstr];
    int i;
    for (i=0; i<noConstr; i++)
    {
    s[i]=0.;
    for (int j=0; j<noCols; j++)
    s[i]+= M[j][i]*x[j];
    }
    for (i=0; i<noConstr; i++)
    cout << i << "," << s[i] << " ";
    cout << endl;
    delete [] s;
    }
    
    
    void Mx(double* x, double* y) const
    {
    long i,j;
    
    for (i = 0; i < noConstr; i++) y[i] = 0.0;
    
    for (j=0; j<noCols; j++) {
    if (x[j] < -MYEPSILON || x[j] > MYEPSILON)
    for (i=0; i<noConstr; i++)
    y[i]+= M[j][i]*x[j];
    }
    
    }
  */
  
  void exportModel(char* filename) 
  {
    cgSolver.exportModel(filename);
  }
};


#endif
