#ifndef TSPTWFIXEDHEGEL_H
#define TSPTWFIXEDHEGEL_H


#include "DATASTRUCTURES/general_includes.h"
#include "DATASTRUCTURES/PERMUTATION/binom.h"
#include "DATASTRUCTURES/ARRAY/array.h"
#include "DATASTRUCTURES/PERMUTATION/permutation.h"
#include "DATASTRUCTURES/SET/set.h"
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <math.h>







class TSPTWFIX   //MINIMIZES!!!
{
    int n; //numVars
    Variable* variables;
    int f;
    SSet fix;

    
    TargetType* thesis;
    TargetType* antithesis;
    TargetType* best;
    TargetType* current;
    TargetType* currentInv;
    double thesisVal;
    double antithesisVal;
    double bestVal;

    double currentVal;
    Eval* e;

    //PARAMETERS
    LearnParam* antiGreedy;
    LearnParam* antiMinSize;
    LearnParam* antiMaxSize;
    LearnParam* antiAltMinSize;
    LearnParam* antiAltMaxSize;
    LearnParam* antiAltProb;
    LearnParam* antiRemProb;
    LearnParam* noImprovementRestart;
    LearnParam* greedyVars;
    LearnParam* restartMinSize;
    LearnParam* restartMaxSize;

    double timelimit;
    double startTime;
    double totalStepsEstimate;
    double totalMovesEstimate;
    double totalRestartsEstimate;

    //double numVars;
    double percentTimeElapsed;
    double restarts;
    double restartsNormed;
    double bestUpdates;
    double bestUpdatesNormed;

    double totalSteps;
    double totalStepsNormed;
    double totalMoves;
    double totalMovesNormed;

    double movesThisRestart;
    double movesThisRestartNormed;
    double movesSinceLastImprovement;
    double movesSinceLastImprovementNormed;
    double movesSinceLastBestUpdate;
    double movesSinceLastBestUpdateNormed;

    double stepsSinceLastBestUpdate;
    double stepsSinceLastBestUpdateNormed;
    double stepsThisRestart;
    double stepsThisRestartNormed;
    double stepsSinceLastImprovement;
    double stepsSinceLastImprovementNormed;


    void order(int& a, int& b)
    {
      if (a<b) return;
      int h=a;
      a=b;
      b=h;
    }

    int getkey(int a, int b, int n)
    {
      order(a,b);
      assert(a<b);
      return a*n - ((a+1)*(a+2))/2  + b;
    }
    
    void updateTime(void)
    {
        double timeElapsed=now()-startTime;
        percentTimeElapsed=minimum(timeElapsed/maximum(0.01,timelimit),1.0);
        if (timeElapsed<0.01)
        {
            totalStepsEstimate=1;
            totalMovesEstimate=1;
            totalRestartsEstimate=1;
            restartsNormed=0;
            bestUpdatesNormed=0;
            totalStepsNormed=0;
            totalMovesNormed=0;
        }
        else
        {
            totalStepsEstimate=maximum(timelimit*totalSteps/timeElapsed,totalSteps);
            totalMovesEstimate=maximum(timelimit*totalMoves/timeElapsed,totalMoves);
            totalRestartsEstimate=maximum(timelimit*restarts/timeElapsed,restarts);
            restartsNormed=restarts/maximum(1,totalRestartsEstimate);
            bestUpdatesNormed=bestUpdates/maximum(1,totalMovesEstimate);
            totalStepsNormed=totalSteps/maximum(1,totalStepsEstimate);
            totalMovesNormed=totalMoves/maximum(1,totalMovesEstimate);
        }
    }


    void restartsInc(void)
    {
        ++restarts;
        updateTime();
    }


    void bestUpdatesInc(void)
    {
        ++bestUpdates;
        updateTime();
    }


    void totalStepsInc(void)
    {
        ++totalSteps;
        updateTime();
    }

    void totalMovesInc(void)
    {
        ++totalMoves;
        updateTime();
    }


    void newRestart(void)
   {
        movesThisRestart=movesThisRestartNormed=0;
        stepsThisRestart=stepsThisRestartNormed=0;
        restartsInc();
    }

    void newImprovement(void)
    {
        movesSinceLastImprovement=movesSinceLastImprovementNormed=0;
        stepsSinceLastImprovement=stepsSinceLastImprovementNormed=0;
    }

    void newBest(void)
    {
        movesSinceLastBestUpdate=movesSinceLastBestUpdateNormed=0;
        stepsSinceLastBestUpdate=stepsSinceLastBestUpdateNormed=0;
        bestUpdatesInc();
    }

    void movesThisRestartInc(void)
    {
        //updateTime();
        ++movesThisRestart;
        if (totalMovesEstimate<=1) movesThisRestartNormed=0;
        else movesThisRestartNormed=movesThisRestart/totalMovesEstimate;
    }

    void movesSinceLastImprovementInc(void)
    {
        //updateTime();
        ++movesSinceLastImprovement;
        if (totalMovesEstimate<=1) movesSinceLastImprovementNormed=0;
        else movesSinceLastImprovementNormed=movesSinceLastImprovement/totalMovesEstimate;
    }

    void movesSinceLastBestUpdateInc(void)
    {
        //updateTime();
        ++movesSinceLastBestUpdate;
        if (totalMovesEstimate<=1) movesSinceLastBestUpdateNormed=0;
        else movesSinceLastBestUpdateNormed=movesSinceLastBestUpdate/totalMovesEstimate;
    }

    void movesInc(void)
    {
        totalMovesInc();
        movesThisRestartInc();
        movesSinceLastImprovementInc();
        movesSinceLastBestUpdateInc();
    }

    void stepsThisRestartInc(void)
    {
        //updateTime();
        ++stepsThisRestart;
        if (totalStepsEstimate<=1) stepsThisRestartNormed=0;
        else stepsThisRestartNormed=stepsThisRestart/totalStepsEstimate;
    }

    void stepsSinceLastImprovementInc(void)
    {
        //updateTime();
        ++stepsSinceLastImprovement;
        if (totalStepsEstimate<=1) stepsSinceLastImprovementNormed=0;
        else stepsSinceLastImprovementNormed=stepsSinceLastImprovement/totalStepsEstimate;
    }

    void stepsSinceLastBestUpdateInc(void)
    {
        //updateTime();
        ++stepsSinceLastBestUpdate;
        if (totalStepsEstimate<=1) stepsSinceLastBestUpdateNormed=0;
        else stepsSinceLastBestUpdateNormed=stepsSinceLastBestUpdate/totalStepsEstimate;
    }

    void stepsInc(void)
    {
        totalStepsInc();
        stepsThisRestartInc();
        stepsSinceLastImprovementInc();
        stepsSinceLastBestUpdateInc();
    }

    void init(void)
    {
      bestUpdates=bestUpdatesNormed=0;
      restarts=restartsNormed=0;
      totalSteps=totalStepsNormed=0;
      totalMoves=totalMovesNormed=0;
      percentTimeElapsed=0;
      movesThisRestart=movesThisRestartNormed=0;
      stepsThisRestart=stepsThisRestartNormed=0;
      movesSinceLastImprovement=movesSinceLastImprovementNormed=0;
      stepsSinceLastImprovement=stepsSinceLastImprovementNormed=0;
      movesSinceLastBestUpdate=movesSinceLastBestUpdateNormed=0;
      stepsSinceLastBestUpdate=stepsSinceLastBestUpdateNormed=0;
    }
    
    void copyThesisToBest(void)
    {
      //assert(bestVal>=thesisVal);
      thesisVal=e->eval(thesis);
      for (int i=0; i<n; i++) 
	best[i]=thesis[i];
      bestVal=thesisVal;
    }
    
    void copyThesisToCurrent(void)
    {
      //cout << endl;
      for (int i=0; i<n; i++) 
	{
	  //  cout << thesis[i] << " ";
	  current[i]=thesis[i];
	  currentInv[current[i]]=i;
	}
      //cout << endl;
      currentVal=thesisVal;
    }
    
    void copyAntiToCurrent(void)
    {
      for (int i=0; i<n; i++) 
	{
	  current[i]=antithesis[i];
	  currentInv[current[i]]=i;
	}
      currentVal=antithesisVal;
    }
    
    void moveThesisToCurrent(void)
    {
      for (int i=0; i<n; i++) 
	thesis[i]=current[i];
      thesisVal=currentVal;
    }
    
    void shiftLeft(int a, int b)
    //leaves b unassigned
    {
      //assert(a<b);
      for (int k=a; k<b; k++)
	{
	  current[k]=current[k+1];            
	}
    }

    void shiftLeftUpdate(int a, int b)
    //leaves b unassigned
    {
      //assert(a<=b);
      for (int k=a; k<b; k++)
	{
	  current[k]=current[k+1];            
	  currentInv[current[k]]=k;
	}
    }

    void shiftRight(int a, int b)
    //leaves a unassigned
    {
      //assert(a<=b);
      for (int k=b; k>a; k--)
	{
	  current[k]=current[k-1];            
	}
    }

    void shiftRightUpdate(int a, int b)
    //leaves a unassigned
    {
      //assert(a<b);
      for (int k=b; k>a; k--)
	{
	  current[k]=current[k-1];            
	  currentInv[current[k]]=k;
	}
    }

    double evalPotentialMove(int i, int j)
    //move item in position i to position j and shift elements in between
    {
      double val;
      TargetType h=current[i];
      if (i<j)
	{
	  shiftLeft(i,j);
	  current[j]=h;
	  val=e->eval(current);
	  shiftRight(i,j);
	  current[i]=h;
	  return val;
	}
      shiftRight(j,i);
      current[j]=h;
      val=e->eval(current);
      shiftLeft(j,i);
      current[i]=h;
      return val;      
    }

    void conductCurrentMove(int i, int j)
    {
      TargetType h=current[i];
      if (i<j)
	{
	  shiftLeftUpdate(i,j);
	  current[j]=h;
	  currentInv[h]=j;
	  return;
	}
      shiftRightUpdate(j,i);
      current[j]=h;
      currentInv[h]=j;
    }


    double evalPotentialSwap(int i, int j)
    {
      double val;
      /*
      std::cout << i << "," << j << "  " << current[i] << "," << current[j] << std::endl;
      std::cout << "EPS Ingoing ";
      printVector(current);
      */
      TargetType h=current[i];
      //std::cout << "h= " << h << endl;
      current[i]=current[j];
      current[j]=h;
      /*
      std::cout << "EPS Evaluating ";
      printVector(current);
      */
      val=e->eval(current);
      current[j]=current[i];      
      current[i]=h;
      /*
      std::cout << "EPS Undoing ";
      printVector(current);
      */
      return val;
    }
    
    void conductCurrentSwap(int i, int j)
    {
      TargetType h=current[i];
      current[i]=current[j];
      current[j]=h;
      currentInv[current[i]]=i;
      currentInv[h]=j;
    }
    
    bool greedyAnti(double percentVars, TargetType* target, bool full=false) //take the current anitthesis and improve greedily
    {
      if (antithesis==target) return false;
      return greedyCore(percentVars,target,true,full);
    }
    
    bool greedy(double percentVars, TargetType* target, bool full=false) //take the current anitthesis and improve greedily
    {
      if (thesis==target) return false;
      return greedyCore(percentVars,target,false,full);
    }
    
    bool greedyCore(double percentVars, TargetType* target, bool anti, bool full) //take the current anitthesis and improve greedily
    //conducts a greedy algorithm by swapping items so values in positions agree more with target assignment
    {
#ifdef VERBOSE
      std::cout << "\n\nstart greedy target ";
      printVector(target);
#endif      
      //      std::cout << "GCT\n";
      int last=-1;
      if (anti)
	{
	  antithesisVal=e->eval(antithesis);
	  copyAntiToCurrent();
	}
      else
	{
	  thesisVal=e->eval(thesis);
	  copyThesisToCurrent();
	}
      /*
      std::cout << "We are currently at ";
      printVector(current);
      printVector(currentInv);
      */
      while (true)
        {
	  //find most improving move
	  //std::cout << "searching new move\n";
	  double minVal;
	  double mostImp=0;
	  double imp;
	  int mostInd=-1;
	  int lastInd1=-1;
	  int lastInd2=-1;
	  TargetType lastVal1,lastVal2;
	  int maxSelect=f; //minimum(n,currentInv[0]+5);
	  int h=minimum(maximum(1,(int)round(maxSelect*percentVars*0.01)),maxSelect);
	  Binom<int> bin(maxSelect,h);
	  bin.select(false);
	  /*
	  std::cout << "selected positions ";
	  for (int k=0; k<h; k++)
	    std::cout << bin[k] << " ";
	  std::cout << std::endl;
	  */

	  for (int s=0; s<h; s++)
            {
	      int i=bin[s];
	      if (i==last || current[i]==target[i]) continue;
	      //std::cout << "considering swap of values " << current[i] << " and " << target[i] << std::endl;
	      double newVal=currentVal;
	      if (lastInd1<0)
		{
		  TargetType h=current[i];
		  if (fabs(target[i]-h)>EPSILON) 
		    {
		      //std::cout << "calling evappotswap\n";
		      newVal=evalPotentialSwap(i,currentInv[target[i]]);
		    }
		  else newVal=e->eval(current);
                }
	      else 
		{
		  newVal=e->eval(i,currentInv[target[i]],lastInd1,lastInd2);
		}
	      lastInd1=i;
	      lastInd2=currentInv[target[i]];
	      if (anti) imp=antithesisVal-newVal;
	      else imp=thesisVal-newVal;
	      if (imp>mostImp)
                {
		  minVal=newVal;
		  mostImp=imp;
		  mostInd=i;
		  if (!full) break;
                }
            }
	  if (mostInd<0) 
	    {
	      //std::cout << "no more swaps found\n";
	      break;
	    }
	  //std::cout << "swapping " << current[mostInd] << " and " << target[mostInd] << std::endl;
	  //take move
	  if (fabs(target[mostInd]-current[mostInd])>EPSILON) conductCurrentSwap(mostInd,currentInv[target[mostInd]]);
	  if (anti) currentVal=antithesisVal=minVal;
	  else currentVal=thesisVal=minVal;
	  /*
	  std::cout << "resulting vector ";
	  printVector(current);
	  */
	  stepsInc();
	  //std::cout << currentVal << "  <>  " << e->eval(current) << "  <>  " << e->eval(thesis) << std::endl << std::flush;
	  last=mostInd;
        }

      if (anti)
	{
	  for (int i=0; i<n; i++)
	    antithesis[i]=current[i];
	  antithesisVal=e->eval(antithesis);
	}
      else
	{
	  for (int i=0; i<n; i++)
	    thesis[i]=current[i];
	  thesisVal=e->eval(thesis);
	}
      /*
      assert(!anti||fabs(antithesisVal-currentVal)<EPSILON);
      assert(!anti||fabs(antithesisVal-e->eval(current))<EPSILON);
      assert(anti||fabs(thesisVal-currentVal)<EPSILON);
      std::cout << currentVal << " <>? " << e->eval(current) << std::endl << std::flush;
      assert(anti||fabs(thesisVal-e->eval(current))<EPSILON);
      */
#ifdef VERBOSE
      if (anti) printVector(antithesis);
      else printVector(thesis);
      printObjectives();
      std::cout << "done with greedy target\n\n\n";
#endif
      return last >= 0;

    }

    void printVector(TargetType* v, bool noNL=false)
    {
      int i=0;
      std::cout << "1 ";
      for (;i<n;i++) 
	{
	  std::cout << e->nodeMapping[v[i]]+1 << " ";
	  if (v[i]==0) break;
	}
      //std::cout << "[" << i << "=" << v[i]<< "] ";
      if (!noNL)
	std::cout << std::endl <<std::flush;
    }


    bool greedyAnti(double percentVars, bool full=false)
    {
      return greedyCore(percentVars,true,full);
    }

    bool greedy(double percentVars, bool full=false)
    {
      return greedyCore(percentVars,false,full);
    }

    bool greedyCore(double percentVars, bool anti, bool full) 
    //conducts a greedy algorithm by taking one var at a time and trying to improve its value
    {
#ifdef VERBOSE
      std::cout << "start greedy coordinate search\n";
#endif
      int last=-1;
      double priorVal;
      if (anti) 
	{
	  copyAntiToCurrent();
	  priorVal=antithesisVal;
	}
      else 
	{
	  copyThesisToCurrent();
	  priorVal=thesisVal;
	}
#ifdef VERBOSE
	  std::cout << "current "; printVector(current);
#endif
	
      while (true)
        {
	  //find most improving move
	  double minVal;
	  double mostImp=EPSILON;
	  double imp;
	  int mostInd1=-1;
	  int mostInd2;
	  int lastInd1=-1;
	  int lastInd2;
	  TargetType lastVal1,lastVal2;
	  int tempEnd=currentInv[0];
	  int maxSelect=f; //n-1;//minimum(n,currentInv[0]+1); //n; //minimum(n,currentInv[0]+5);
	  //percentVars*=0.01*myRand(67,100);
	  int h=minimum(maximum(1,(int)round(maxSelect*percentVars*0.01)),maxSelect);
	  Binom<int> bin(maxSelect,h);
	  /*
	  Binom<int> bin(n,h);
	  int h=minimum(maximum(1,(int)round(n*percentVars*0.01)),n);
	  */
	  bin.select(true);
	  SSet selected(maxSelect+1);
	  for (int s=0; s<h; s++)
            {
	      int i=bin[s];
	      selected.add(i);
	      //since new subset we may actually want to consider last after all: if (i==last) continue;

	      for (int j=0; j<f; j++) //(int t=s+1; t<h; t++)
		{
		  if (selected.isIn(j)) continue;		  
		  double newVal=currentVal;
		  if (lastInd1<0) newVal=evalPotentialSwap(i,j);
		  else
		    {
		      newVal=e->eval(i,j,lastInd1,lastInd2);
		    }
		  lastInd1=i;
		  lastInd2=j;
		  imp=priorVal-currentVal;
		  if (imp>mostImp)
		    {
		      minVal=newVal;
		      mostImp=imp;
		      mostInd1=i;
		      mostInd2=j;
		      if (!full) goto IMPDONE;
		    }
		}	      
	    }

	IMPDONE:
	  if (mostInd1<0) break;
	  //take move
	  conductCurrentSwap(mostInd1,mostInd2);
	  currentVal=priorVal=minVal;
	  stepsInc();
	  //std::cout << currentVal << "  <>  " << e->eval(current) << "  <>  " << e->eval(antithesis) << std::endl << std::flush;
	  //last=mostInd1;
        }
      if (anti)
	{
	  for (int i=0; i<n; i++)
	    antithesis[i]=current[i];
	  antithesisVal=currentVal;
	}
      else
	{
	  for (int i=0; i<n; i++)
	    thesis[i]=current[i];
	  thesisVal=currentVal;
	}
    
#ifdef VERBOSE
      if (anti) printVector(antithesis);
      else printVector(thesis);
      printObjectives();
      std::cout << "done with greedy coordinate search\n";
#endif
      return last >= 0;
    }


    void throwAntithesis(double minPercent, double maxPercent, bool full=false)
    {
#ifdef VERBOSE
      std::cout << "throw antithesis\n";
#endif
      throwRandomization(minPercent,maxPercent,true,full);
    }
    
    void throwRestart(double minPercent, double maxPercent, bool full=false)
    {
      static int lastIndex=-1;
      throwRandomization(minPercent,maxPercent,false,full);
    }

    void throwRandomization(double minPercent, double maxPercent, bool anti, bool full)
    {
      TargetType* vector;
      if (anti) 
	{
	  vector=antithesis;
	  for (int i=0; i<n; i++)
	    {
	      current[i]=vector[i]=thesis[i];
	      if (current[i]==0) 
		{
		  assert(i==f);
		}
	    }
	}
      else 
	{
	  vector=thesis;
	  for (int i=0; i<n; i++)
	    {
	      current[i]=thesis[i];
	      if (current[i]==0)
		{
		  assert(i==f);
		}
	    }
	}

      if (maxPercent<minPercent) maxPercent=minPercent;
      int maxSelect=maximum(f,1);
      int s=myRand(maximum(minimum(maxSelect,(int)round(maxSelect*minPercent*0.01)),1),maximum(minimum((int)round(maxSelect*maxPercent*0.01),maxSelect),1));
      assert(current[f]==0);
      assert(maxSelect>=1);
      /*
      Binom<int> pick(maxSelect,s);
      pick.select(false);
      Permutation<int> perm(s);
      perm.permute();
      */
      int pick[s];
      int perm[s];
      int vals[s];
      
      for (int i=0,j=myRand(0,maximum(f-1,0)); i<s; i++) 
	{
	  assert(j>=0 && j<n);
	  //	  std::cout << j << " " << std::flush;
	  pick[i]=j;
	  assert(current[j]<n);
	  assert(current[j]>=0);
	  assert (current[j]>0);
	  vals[i]=myRand(e->nodeStart[current[j]],e->nodeEnd[current[j]]);
	  j=(f+j-1)%f;
	}
      radixSort(vals,s,3,&perm[0]);

      for (int k=0; k<s; k++)
	{
	  int i=pick[k];
	  current[i]=vector[pick[perm[k]]];
	}
      for (int k=0; k<s; k++)
	{
	  int i=pick[k];
	  vector[i]=current[i];
	}

      updateTime();
      if (anti)
	{
	  antithesisVal=e->eval(antithesis);
	  if (myRand(1,100)<=antiGreedy->getPercentage())
	    greedyAnti(greedyVars->getPercentage(),full);
	}
      else thesisVal=e->eval(thesis);
#ifdef VERBOSE
      printVector(vector);
      printObjectives();
      std::cout << "antithesis is thrown\n";
#endif
    }
	

    int move(void)
    {
      double oldVal=thesisVal;
      bool moved=greedy(100, antithesis,false);
      if (thesisVal<oldVal-EPSILON) return 1;
      if (moved) return -1;
      return 0;
    }


    /*
    int move(void)
    {
#ifdef VERBOSE
      std::cout << "move discretely between thesis and antithesis\n";
        //std::cout << "move from thesis " << thesisVal << " to antiThesis " << e->eval(antithesis) << std::endl;
      std::cout << "thesis "; printVector(thesis);
      std::cout << "antithesis "; printVector(antithesis);
      printObjectives();
#endif
        copyThesisToCurrent();
        SSet candidates(n);
	int deltaInd[2*n];
	int moveInd[2*n];
	TargetType deltaVal[2*n];
	TargetType moveVal[2*n];
	int deltaSize=0;
	int moveSize=0;
	
        for (int i=0; i<n; i++)
	  {
	    if (fabs(thesis[i]-antithesis[i])>EPSILON) candidates.add(i);
	  }
        double moveValue=thesisVal;
	SSet cache((n*(n-1))/2);
        while (candidates.card>0)
	  {
#ifdef VERBOSE
	    std::cout << "current "; printVector(current);
#endif
	    double minVal;
	    double mostImp=0;
	    double imp;
	    int mostInd=-1;
	    TargetType mostVal;
	    Permutation<int> perm(candidates.card);
	    perm.permute();
	    int lastInd1=-1;
	    int lastInd2=-1;
	    TargetType lastVal1,lastVal2;
	    for (int k=0; k<candidates.card; k++)
	      {
		int i=candidates.list[perm[k]];
#ifdef VERBOSE
		std::cout << "considering var " << i << std::endl;
#endif
		if (current[i]==antithesis[i]) continue;
		int key = getkey(i,currentInv[antithesis[i]],n);
		if (cache.isIn(key)) continue;
		else cache.add(key);
		//cout << currentVal << std::endl << std::flush;

		double candidateVal=currentVal;
		if (lastInd1<0) 
		  {
		    //std::cout << "pot swap " << i << " " << currentInv[antithesis[i]] << std::endl << std::flush;
		    candidateVal=evalPotentialSwap(i,currentInv[antithesis[i]]);
		    //cout << candidateVal << std::endl << std::flush;
		  }
		else 
		  {
		    //std::cout << "eval " << i << " " << currentInv[antithesis[i]] << " " << lastInd1 << " " << lastInd2 << std::endl << std::flush;
		    candidateVal = e->eval(i,currentInv[antithesis[i]],lastInd1,lastInd2);
		    //cout << candidateVal << std::endl << std::flush;
		  }
		lastInd1=i;
		lastInd2=currentInv[antithesis[i]];
		//cout << candidateVal << " <> " << flush;
		//cout << currentVal << flush;
		imp=currentVal-candidateVal;
		if ((mostInd<0)
		    ||(imp>mostImp))
		  {
		    minVal=candidateVal;
		    mostImp=imp;
		    mostInd=i;
		    mostVal=antithesis[i];
		    if (imp>0) break;   //SHOULD BE PARAMETERIZED!
		  }
	      }
	    if (mostInd<0) break;
	    assert(mostInd>=0);
	    stepsInc();
	    candidates.rem(mostInd);

	    //take the move
	    TargetType h=current[mostInd];
	    currentVal=minVal;
	    deltaInd[deltaSize]=mostInd;
	    deltaVal[deltaSize++]=mostVal;
	    deltaInd[deltaSize]=currentInv[mostVal];
	    deltaVal[deltaSize++]=h;
	    conductCurrentSwap(mostInd,currentInv[mostVal]);
	    if (currentVal<=moveValue)
	      {
		for (int r=0; r<deltaSize; r++)
		  {
		    moveInd[moveSize]=deltaInd[r];
		    moveVal[moveSize++]=deltaVal[r];
		  }
		deltaSize=0;
		moveValue=currentVal;
	      }
	    cache.remAll();
	  }
        movesInc();
        if (moveSize==0) 
	  {
#ifdef VERBOSE
	    std::cout << "move was unsuccessful\n";
#endif
	    return 0;
	  }
        //std::cout << "\nmoving -> " << moveValue << std::endl;
        for (int r=0; r<moveSize; r++)
	  thesis[moveInd[r]]=moveVal[r];
#ifdef VERBOSE
	printVector(thesis);
	printObjectives();
#endif
        if (fabs(thesisVal-moveValue)<EPSILON) 
	  {
	    thesisVal=moveValue;
#ifdef VERBOSE
	    std::cout<< "move did not improve objective\n";
#endif
	    return -1;
	  }
        thesisVal=moveValue;
        //if (thesisVal<antithesisVal) std::cout << "X";
        //std::cout << thesisVal << " <>  " << e->eval(thesis) << std::endl;
#ifdef VERBOSE
	std::cout << "move improved objective\n";
#endif
        return 1;
    }
    */


public:

 TSPTWFIX(int _f, int* _fix, int _n, Variable*& v, Eval* _e, LearnParam* _gV, LearnParam* _aG, LearnParam* _aMin, LearnParam* _aMax, LearnParam* _nI, LearnParam* _rMin, LearnParam* _rMax)
   : f(_f), n(_n), e(_e), fix(_n), antiGreedy(_aG), antiMinSize(_aMin), antiMaxSize(_aMax), noImprovementRestart(_nI), greedyVars(_gV), restartMinSize (_rMin), restartMaxSize (_rMax)
      {
	//std::cout << "FIX VARIANT\n";
	for (int i=0; i<f; i++)
	  fix.add(_fix[f]);
	e->eval(_fix);

	antithesisVal=-999;
	variables=v;
	//v=0;
	thesis = new TargetType[n];
	antithesis = new TargetType[n];
	best = new TargetType[n];
	current = new TargetType[n];
	currentInv = new TargetType[n];

	/*
        double** features=new double*[11];
        features[0]=&percentTimeElapsed;
        features[1]=&restartsNormed;
        features[2]=&bestUpdatesNormed;
        features[3]=&totalStepsNormed;
        features[4]=&totalMovesNormed;
        features[5]=&movesThisRestartNormed;
        features[6]=&movesSinceLastImprovementNormed;
        features[7]=&movesSinceLastBestUpdateNormed;
        features[8]=&stepsSinceLastBestUpdateNormed;
        features[9]=&stepsThisRestartNormed;
        features[10]=&stepsSinceLastImprovementNormed;
	
        antiGreedy->setFeatures(features);
        antiMinSize->setFeatures(features);
        antiMaxSize->setFeatures(features);
        noImprovementRestart->setFeatures(features);
        greedyVars->setFeatures(features);
        restartMinSize->setFeatures(features);
        restartMaxSize->setFeatures(features);
	
        delete [] features;
	*/
    };
    
    ~TSPTWFIX()
    {
        delete [] thesis;
        delete [] antithesis;
        delete [] best;
        delete [] current;
	delete [] currentInv;
	//delete [] variables;
    }

    void recordBest(int* x)
    {
      for (int i=0; i<n; i++) 
	{
	  x[i]=best[i];
        }
    }

    void printObjectives(void)
    {
      std::cout << "thesis value -> " << thesisVal << "  antithesis value -> " << antithesisVal << "  best value -> " << bestVal << std::endl << std::flush; 
    }

    double minimize(double tl, double lb=-10000000000)
    {
      /*
      int perm[10];
      int nums[10]={3,2,1,0,4,5,6,7,8,9};
      for (int i=0; i<10; i++)
	std::cout << nums[i] << " ";
      std::cout << std::endl;
      radixSort(nums,10,1,perm);
      for (int i=0; i<10; i++)
	std::cout << nums[i] << " ";
      std::cout << std::endl;
      for (int i=0; i<10; i++)
	std::cout << perm[i] << " ";
      std::cout << std::endl;
      */

      static int listening=1;
      init();
      timelimit=tl;
      startTime=now();
      //mySRand(212329);
      for (int i=0; i<n; i++)
        best[i]=thesis[i]=e->getValue(i); //(bool)myRand(0,1);
	/*
        for (int i=0; i<n; i++)
	  std::cout << thesis[i] << " ";
	std::cout << endl;
	*/
      bestVal=thesisVal=e->eval(thesis);
      /*
      std::cout << "ingoing vector: ";
      printVector(thesis);
      */
      assert(std::isfinite(bestVal));
        //std::cout << "starting greedy\n" << std::flush;
      updateTime();
      greedy(greedyVars->getPercentage());
      
      //std::cout << "after initial greedy: ";

      copyThesisToBest();
      bestVal=thesisVal;
      //printVector(thesis);
      if (bestVal<=lb) goto FINISHED;
      double rsp;
      //std::cout << "starting search\n";

      while (true)
	{
	  //printVector(thesis);
	  updateTime();
	  //std::cout << "\n---RRR---\n";
	  int noImpCounter=0;
	  do
	    {
#ifdef VERBOSE
		std::cout << "\nNew Dialective Step\nCurrent time -> " << now() << std::endl << std::flush;
		printObjectives();
#endif
                if(now()-startTime>timelimit) goto TIMEOUT;
#ifdef VERBOSE
		std::cout<<"Throw Antithesis\n" << std::flush;
#endif
		//		cout << "\n\nTESTING NEW ANTITHESIS\n";
		int m;
		throwAntithesis(antiMinSize->getPercentage(),antiMaxSize->getPercentage());
		/*
		std::cout << "antithesis: ";
		printVector(antithesis);
		*/
		m=move(); //path relink by flipping vars from thesis to antithesis value
		/*
		std::cout << "after move: ";
		printVector(thesis);
		*/
                if (m!=0)
		  {		    
		    //std::cout << "We moved the thesis!\n" << std::flush;
                    if (m>0) 
		      {
			//std::cout << "Improved it even\n";
			newImprovement();
			noImpCounter=0;
		      }
		    //else std::cout << "+";
                    updateTime();
		    //std::cout << "Greey attempt 1!\n" << std::flush;
                    greedy(greedyVars->getPercentage());
		    assert(isfinite(thesisVal));
		    //greedy(greedyVars->getPercentage(),thesis);
		    //std::cout << "Greey attempt 2!\n" << std::flush;
		    greedy(greedyVars->getPercentage(),antithesis);
		    /*
		    std::cout << "resulting vector: ";
		    printVector(thesis);
		    */
		    assert(std::isfinite(thesisVal));		    
                    if (thesisVal<bestVal)
		      {
			bestVal=thesisVal;
			assert(std::isfinite(thesisVal));
			copyThesisToBest();
			if (bestVal<=lb) goto FINISHED;
			newBest();
		      }
		  }
		else 
		  {
		    //std::cout << "no improvement";
		  }
                updateTime();
#ifndef FIXFORCERESTARTS
		rsp=noImprovementRestart->getPercentage();
#else
		rsp=myRand(1,10)*0.01;
#endif
		//		std::cout << "RES PROB -> " << rsp << "  NoImC -> " << noImpCounter << "  Listen -> " << listening << std::endl;
	      }
	    while (myRand(0,10000)>=100*rsp);
            //move to new point
#ifdef VERBOSE
	    std::cout << "\n\nNew Restart!\n";
#endif
	    double minSize=restartMinSize->getPercentage();
	    double maxSize=restartMaxSize->getPercentage();
	    if (maxSize<minSize) maxSize=minSize;
	    throwRestart(minSize,maxSize);
	    newRestart();
	    thesisVal=e->eval(thesis);
	    
            greedy(greedyVars->getPercentage());
            if (thesisVal<bestVal)
            {
	      bestVal=thesisVal;
	      copyThesisToBest();
	      if (bestVal<=lb) goto FINISHED;
	      newBest();
            }
        }
    FINISHED:
    TIMEOUT:
      //      std::cout << bestVal << std::endl;
      return bestVal;
    }

};

#endif
