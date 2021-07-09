//// g++ -c -std=c++0x -O3 -DNDEBUG -I../boost_1_63_0/ -IDATASTRUCTURES/ cont-hegel.cpp ;  g++ -c -std=c++0x -O3 -DNDEBUG -I../boost_1_63_0/ -IDATASTRUCTURES/ DATASTRUCTURES/general_includes.cpp ;  g++ -o search -std=c++0x -O3 -DNDEBUG -I../boost_1_63_0/ -IDATASTRUCTURES/ cont-hegel.o general_includes.o



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
#include "ARRAY/array.h"

#ifdef EXPENSIVE
#include "tsptw-hegel-exp.h"
#else
#include "tsptw-hegel.h"
#endif





class TSPTW: public Eval
{
  int n; //number of nodes
  int k; //number of scenarios
  int shortK;
  bool kChange;
  int* selection;
  //  Binom<int>* scenSelector;

  double*** arcTime;
  double tourPenalty;
  double nodePenalty;
  double* nodeReward;
  int origN;



  int* tourP;
  int tourPEnd;
  double** nodeDepartProbs;
  double* currentArrivalDistribution;
  double tourPCost;
  double* nodePCost;
  bool modifiedObjective;
  double* nodePMinorObj;
  double tourPMinorObj;
  double* minorPNorm;

  
  int* tourS;
  int tourSEnd;
  double tourSCost;
  double** nodeSCost;
  double** nodeSTime;
  double* nodeSMinorObj;
  double tourSMinorObj;
  double* minorSNorm;
  
  
  SSet* minorSelection;
  double minorPolyExp;
  double minorCoeff;
  double* minorNormalizer;
  double* minorTimePenalty;
  
  
  double initMinorTimePenalty(void)
  {
    for (int ti=0; ti<=maxTime; ti++)
      minorTimePenalty[ti]=exp(minorPolyExp*log(maximum(ti,1)));
  }
  
  double simulateScen(int oldNode, int start, bool full=false)
  {
    return simulateScenario(oldNode,start,full);
  }
  
  double simulateProb(int oldNode, int start, bool full=false)
  {
    return simulateProbability(oldNode,start,full);
  }
  
  
  
  double simulateScenario(int oldNode, int start, bool full)
  {
    int numScen=minimum(k,shortK);
    if (kChange)
      {
	selection[0]=0;
	if (numScen>1)
	  {
	    //cout << k-2 << " / " << numScen-1 <<endl;
	    //Binom<int> scenSelector(k-2,numScen-1);
	    //scenSelector.select();
	    for (int h=0; h<numScen-1; h++)
	      selection[h+1]=h+1; //scenSelector[h]+1;
	  }
	kChange=false;
	oldNode=0;
	start=0;
      }
    if (full) 
      {
	oldNode=0;
	start=0;
	numScen=k;
      }
    
    
    if (start>tourSEnd) 
      {
	if (modifiedObjective)
	  return tourSCost-minorCoeff*tourSMinorObj;    
	else
	  return tourSCost;
      }
    tourSCost=0;
    
    
    double minorValue=0;
    double minorNorm=0;
    if (modifiedObjective)
      {
	if (oldNode>0) 
	  {
	    minorValue=nodeSMinorObj[oldNode];
	    //	std::cout << minorValue << std::endl;
	    minorNorm=minorSNorm[oldNode];
	  }
	if (tourS[start]>0) 
	  {
	    assert(tourS[start]<n);
	    if (minorSelection->isIn(tourS[start])) 
	      {
		nodeSMinorObj[tourS[start]]=n-start+1+minorValue;
	      }
	    else nodeSMinorObj[tourS[start]]=minorValue;
	  }
	else assert(start==tourSEnd);
	for (int i=start+1; i<tourSEnd; i++)
	  {      
	    int currentNode=tourS[i];
	    if (minorSelection->isIn(currentNode)) 
	      {
		nodeSMinorObj[currentNode]=n-i+1+nodeSMinorObj[tourS[i-1]];
	      }
	    else nodeSMinorObj[currentNode]=nodeSMinorObj[tourS[i-1]];
	  }    
      }
    double penalties=0;
    
    int lastNode=oldNode;
    for (int i=start; i<=tourSEnd; i++)
      {      
	int currentNode=tourS[i];
	assert(i<tourSEnd || currentNode==0);
	assert(lastNode>=0&&lastNode<n);
	assert(currentNode>=0&&currentNode<n);
	for (int sh=0; sh<numScen; sh++)
	  {
	    int s=sh; 
	    if (!full) s=selection[sh];
	    assert(s<k);
	    assert(nodeSTime[s][0]<EPSILON);
	    assert(nodeSCost[s][0]<EPSILON);
	    double currentTime=nodeSTime[s][lastNode];
	    double newCost=nodeSCost[s][lastNode];
	    
	    //assert(s>=0 && s<numScen);
	    assert(s>=0 && s<k);
	    currentTime+=arcTime[s][lastNode][currentNode];
	    if (currentNode==0) 
	      {
		assert(i==tourSEnd);
		if (currentTime>maxTime) newCost+=tourPenalty;
		else
		  {
		    if (modifiedObjective)
		      {
			if (minorSelection->isIn(currentNode))
			  {
			    double pen=minorTimePenalty[minimum(maxTime,(int)round(currentTime))]/(minorTimePenalty[maxTime]*numScen);
			    //			 std::cout << pen << std::endl;
			    penalties+=(n-i+1)*pen;
			  }
		      }
		  }
		tourSCost+=newCost;
		continue;
	      }
	    
	    if (currentTime>nodeEnd[currentNode]) newCost+=nodePenalty;
	    else 
	      {
		if (modifiedObjective)
		  {
		    if (minorSelection->isIn(currentNode))
		      {
			double pen=minorTimePenalty[minimum((int)round(currentTime),maxTime)]/(minorNormalizer[currentNode]*numScen);
			//			 std::cout << pen << std::endl;
			penalties+=(n-i+1)*pen;
		      }
		  }
		if (currentTime<nodeStart[currentNode]) currentTime=nodeStart[currentNode];
		newCost-=nodeReward[currentNode];
	      }
	    nodeSTime[s][currentNode]=currentTime;
	    nodeSCost[s][currentNode]=newCost;
	    assert(fabs(nodeSTime[s][0]<EPSILON));
	    assert(fabs(nodeSCost[s][0]<EPSILON));	      
          }
	if (modifiedObjective)
	  {
	    if (minorSelection->isIn(currentNode))
	      minorNorm+=n-i+1;
	    
	    if (currentNode>0) 
	      {
		nodeSMinorObj[currentNode]-=penalties;
		minorSNorm[currentNode]=minorNorm;	     
	      }
	  }
	lastNode=currentNode;
      }
    tourSCost/=numScen;
    
    if (modifiedObjective)
      {
        if (minorNorm>0)
	  {
	    //	std::cout << minorNorm << std::endl;
	    //	std::cout << penalties << std::endl;
	    minorValue+=minorNorm-penalties;
	    //	std::cout << minorValue << std::endl;
	    tourSMinorObj=minorValue/minorNorm;
	  }
	else
	  {
	    minorValue=0;
	    tourSMinorObj=0;
	  }
	//    std::cout << tourSMinorObj << std::endl;
	return tourSCost-minorCoeff*tourSMinorObj;
      }
    else return tourSCost;
  }

  double simulateProbability(int oldNode, int start, bool full)
  {
    /*
    std::cout << "Sim prob\n";
    for (int h=0; h<n; h++) std::cout << tourP[h] << " ";
    std::cout << std::endl;
    */
    if (full||kChange) 
      {
	oldNode=0;
	start=0;
      }

    double minorValue=0;
    double minorNorm=0;
    if (modifiedObjective)
      {
        if (oldNode>0) 
	  {
             minorValue=nodePMinorObj[oldNode];
             minorNorm=minorPNorm[oldNode];
          }
        assert(oldNode>0||nodePMinorObj[oldNode]<EPSILON);
      }

    if (start>tourPEnd) 
      {
	if (modifiedObjective)
	  {
	/*
	std::cout << tourPCost << " - " 
		  << minorCoeff << " * " 
		  << tourPMinorObj << std::endl;
	*/
	    return tourPCost-minorCoeff*tourPMinorObj;
	  }
	else return tourPCost;
      }
    tourPCost=nodePCost[oldNode];
    assert(oldNode>0||(fabs(nodeDepartProbs[0][0]-1)<EPSILON&&tourPCost<EPSILON));

    for (int i=start; i<=tourPEnd; i++)
      {      
	int currentNode=tourP[i];
	assert(oldNode>=0&&oldNode<n);
	assert(currentNode>=0&&currentNode<n);
	/*	
	std::cout << "(" << oldNode << "," << currentNode << ") ->  " << arcTime[k+1][oldNode][currentNode] << std::endl
		  << "Depart -> ";
	for (int h=0; h<=maxTime+1; h++) cout << nodeDepartProbs[oldNode][h] << " ";
	std::cout << std::endl;
	*/
	if (arcTime[k+1][oldNode][currentNode]<EPSILON)
	  {
	    //	    std::cout << "Zero arc\n";
	    copy(maxTime+2,currentArrivalDistribution,nodeDepartProbs[oldNode]);
	  }
	else
	  {
	    for (int ti=0; ti<=maxTime+1; ti++)
	      currentArrivalDistribution[ti]=0;
	    for (int sti=0; sti<maxTime; sti++) 
	      {
		double uniformProb=nodeDepartProbs[oldNode][sti];
		if (uniformProb<EPSILON) continue;
		uniformProb=uniformProb*1./arcTime[k+1][oldNode][currentNode];
		for (int ati=sti+1; ati<=minimum(maxTime,sti+arcTime[k+1][oldNode][currentNode]); ati++) 
		  currentArrivalDistribution[ati]+=uniformProb;
	      }
	  }
	/*
	std::cout << "Arrive -> ";
	for (int h=0; h<=maxTime+1; h++) cout << currentArrivalDistribution[h] << " ";
	std::cout << std::endl;

	std::cout << "Current Node -> " << currentNode << ": " << nodeStart[currentNode] << "-" << nodeEnd[currentNode] << std::endl;
	*/
	double nodeTimePenalty=0;
	double startMass=0;
	for (int ti=0; ti<nodeStart[currentNode]; ti++)
	  {
	    startMass+=currentArrivalDistribution[ti];

	    if (modifiedObjective) nodeTimePenalty+=minorTimePenalty[ti]*currentArrivalDistribution[ti];
	    currentArrivalDistribution[ti]=0;
	  }
	startMass+=currentArrivalDistribution[nodeStart[currentNode]];
	if (modifiedObjective) if (nodeStart[currentNode]<=maxTime) nodeTimePenalty+=minorTimePenalty[nodeStart[currentNode]]*currentArrivalDistribution[nodeStart[currentNode]];

	
	currentArrivalDistribution[nodeStart[currentNode]]=startMass;
	for (int ti=nodeStart[currentNode]+1; ti<=nodeEnd[currentNode]; ti++)
	  {
	    startMass+=currentArrivalDistribution[ti];
	    if (modifiedObjective) if (ti<=maxTime) nodeTimePenalty+=minorTimePenalty[ti]*currentArrivalDistribution[ti];
	  }
	if (modifiedObjective)
          { 
	    nodeTimePenalty/=minorNormalizer[currentNode];
	    if (minorSelection->isIn(currentNode)) 
	      {
		minorValue+=(n-i+1)*(1-maximum(minimum(nodeTimePenalty,1),0));
		minorNorm+=n-i+1;
	      }
	    assert(nodeTimePenalty>-EPSILON&&nodeTimePenalty<1+EPSILON);
	  }
	startMass=minimum(maximum(startMass,0.0),1.0);
	tourPCost+=(1-startMass)*nodePenalty-startMass*nodeReward[currentNode];
	for (int ti=nodeEnd[currentNode]+1; ti<=maxTime; ti++)
	  {
	    startMass+=currentArrivalDistribution[ti];
	  }
	assert(startMass<=1+EPSILON);
	startMass=maximum(minimum(startMass,1.),0.);
	currentArrivalDistribution[maxTime+1]=1-startMass;
	/*
	std::cout << "Corrected Arrive -> ";
	for (int h=0; h<=maxTime+1; h++) cout << currentArrivalDistribution[h] << " ";
	std::cout << std::endl;
	*/
	if (currentNode==0) 
	  {
	    assert(i==tourPEnd);
	    break;
	  }
	if (modifiedObjective)
	  {
	    assert(currentNode>0);
	    nodePMinorObj[currentNode]=minorValue;
	    minorPNorm[currentNode]=minorNorm;
          }
	copy(maxTime+2,nodeDepartProbs[currentNode],currentArrivalDistribution);
	assert(currentNode>0);
	nodePCost[currentNode]=tourPCost;
	oldNode=currentNode;
      }
    tourPCost+=currentArrivalDistribution[maxTime+1]*tourPenalty;    
    if (modifiedObjective)
      {
	if (minorNorm>0)
	  minorValue/=minorNorm;
	else minorValue=0;
	//std::cout << minorValue << std::endl;
	assert(minorValue<=1+EPSILON && minorValue>-EPSILON);
	tourPMinorObj=maximum(minimum(minorValue,1),0);
	/*
	  std::cout << tourPCost << " - " 
	  << minorCoeff << " * " 
	  << tourPMinorObj << std::endl;
	*/
	return tourPCost-minorCoeff*tourPMinorObj;
      }
    else return tourPCost;
  }

  void swap(int* t, int& te, int a, int b)
  {
    if (a==b) return;
    int h=t[a];
    t[a]=t[b];
    t[b]=h;
    if (t[a]==0) te=a;
    else if (t[b]==0) te=b;
  }


  void analyze(void)
  {
    for (int i=0; i<n-1; i++)
      for (int j=i+1; j<n; j++)
	{
	  std::cout << "Analyzing pair (" << i << "," << j << "): ";
	  if (nodeStart[i]+arcTime[0][i][j]>nodeEnd[j] && nodeStart[j]+arcTime[0][j][i]>nodeEnd[i])
	    std::cout << "incompatible\n";
	  else std::cout << "ok\n";
	}
  }





public:

  void setReducedScenarioNumber(double f)
  {
    shortK=maximum(1,round(k*f));
    //std::cout << "new short k -> " << shortK << std::endl;
    int c=n/2;
    Binom<int> sel(n,c);
    if (modifiedObjective)
      {
	minorSelection->remAll();
	for (int i=0; i<c; i++)
	  minorSelection->add(sel[i]);
      }
    kChange=true;
  }


  int getTourEnd(void)
  {
    return tourSEnd;
  }

  int getValue(int i)
  {
    return tourS[i];
  }

  int getNumberOfNodes(void)
  {
    return n;
  }

  void printCurrent(void)
  {
    std::cout << 1 << " ";
    for (int i=0; i<n; i++)
      {
	std::cout << tourS[i]+1 << " ";
	if (tourS[i]==0) std::cout << "| ";
      }
    cout << std::endl;
  }

TSPTW(int _n, int _k, char* _nodeFile, char* _arcFile, double _nodePenalty, int _maxTime, double _tourPenalty, double _polyExp, bool _modObj) : 
  n(_n), k(_k), nodePenalty(_nodePenalty), tourPenalty(_tourPenalty), kChange(true), minorCoeff(0.01), minorPolyExp(_polyExp), modifiedObjective(_modObj)
  {
    nodeStart=new int[n];
    nodeEnd=new int[n];
    nodeReward=new double[n];
    selection=new int[k+2];

    nodeMapping=new int[n];
    arcTime=new double**[k+2];
    nodeSCost=new double*[k+2];
    nodeSTime=new double*[k+2];
    nodeDepartProbs=new double*[n];
    nodePCost=new double[n];
    nodePMinorObj=new double[n];
    nodePMinorObj[0]=0;
    tourPMinorObj=0;
    minorSNorm=new double[n];
    minorPNorm=new double[n];
    nodeSMinorObj=new double[n];
    nodeSMinorObj[0]=0;
    tourSMinorObj=0;
    
    double maxArcTime[n][n];
    string temp;
    double expArcTime[n][n];
    ifstream expStream( _arcFile );
    int id=0;
    while( getline( expStream, temp ) )
      {
	vector<string> elems;
	splitString(temp, ',', elems);
	if (elems[0][0]=='#') continue;
	if (elems.size()!=(unsigned)n)
	  {
	    //cout << "ARC INPUT FILE ERROR: LINE " << numberOfLines << endl;
	    exit (999);
	  }	
	for (int i=id; i<n; i++)
	  {
	    assert((double)atof(elems[i].c_str())>=0);
	    expArcTime[i][id]=expArcTime[id][i]=(double)atof(elems[i].c_str()); //*0.9; ///**0.505;
	  }
	id++;
      }
    expStream.close();

    for (int s=0; s<k+2; s++)
      {
	nodeSCost[s]=new double[n];
	nodeSTime[s]=new double[n];
	nodeSCost[s][0]=nodeSTime[s][0]=0;
	//cout << "nodeSTime[" << s << "][0] = " << nodeSTime[s][0] << endl << flush;
	arcTime[s]=new double*[n];
	for (int i=0; i<n; i++)
	  arcTime[s][i]=new double[n];
      }

    map<string,int> nodeId;
    map<string,int> checkId;
    ifstream nodeStream( _nodeFile );
    int numberOfLines=0;
    SSet ids(n);
    SSet unreachable(n); 
    double maxReward=0;

    while( getline( nodeStream, temp ) )
      {
	numberOfLines++;
	vector<string> elems;
	splitString(temp, ',', elems);
	if (elems[0][0]=='#') continue;
	if (elems.size()!=7)
	  {
	    cout << "NODE INPUT FILE ERROR: LINE " << numberOfLines << "  -> Wrong NUMBER OF ENTRIES\n";
	    exit (999);
	  }
	int id=atoi(elems[0].c_str())-1;
	if ((id<0) || (id>=n))
	  {
	    cout << "NODE INPUT FILE ERROR: LINE " << numberOfLines << "  -> WRONG NODE ID NUMBER\n";
	    exit (999);
	  }
	if (ids.isIn(id))
	  {
	    cout << "NODE INPUT FILE ERROR: LINE " << numberOfLines << "  -> DOUBLE ENTRY FOR NODE " << id+1 << endl;
	    exit (999);
	  }
	ids.add(id);
	nodeStart[id]=atoi(elems[3].c_str());
	nodeEnd[id]=atoi(elems[4].c_str());
	nodeReward[id]=(double)atof(elems[5].c_str());
	if (nodeReward[id]<0)
	  {
	    cout << "NODE INPUT FILE ERROR: LINE " << numberOfLines << "  -> NEGATIVE NODE REWARD\n";
	    exit (999);
	  }
	if (ids.card==1) maxTime=atoi(elems[6].c_str());
	else if (fabs(maxTime-(double)atof(elems[6].c_str())) > EPSILON)
	  {
	    cout << "NODE INPUT FILE ERROR: LINE " << numberOfLines << "  -> INCONSISTENT MAX TOUR TIME\n";
	    exit (999);	    
	  }
	//cout << id+1 << " in [" << nodeStart[id] << "," << nodeEnd[id] << "] for " << nodeReward[id] << endl; 
      }
    nodeStream.close();
    assert(ids.card<=n);
    if (ids.card!=n)
      {
	cout << "NODE INPUT FILE ERROR: LINE " << numberOfLines << "  -> TOO FEW LINES\n";
	exit (999);	    
      }
    if (_maxTime>0 && maxTime!=_maxTime)
      {
	maxTime=_maxTime;
	cout << "OVERRIDING TOUR TIME LIMIT TO " << maxTime << endl;
      }

    for (int i=0; i<n; i++)
      nodeEnd[i]=minimum(nodeEnd[i],maxTime);

    for (int id=0,i=0; i<n; i++)
      {
#ifdef PRUNE
	if (nodeStart[i]+0.505*expArcTime[i][0]>maxTime || 2*0.505*expArcTime[i][0]>maxTime || 0.505*expArcTime[0][i]>nodeEnd[i]) 
	  {
	    unreachable.add(i);
	    continue;
	  }
	/*
	if (nodeStart[i]+expArcTime[i][0]>maxTime || 2*expArcTime[i][0]>maxTime)
	  {
	    std::cout << "Node " << i << " with reward " << nodeReward[i] << " is questionable!\n"; 
	  }
	*/
#endif
	nodeDepartProbs[id]=new double[maxTime+2];
	nodeStart[id]=nodeStart[i];
	nodeEnd[id]=minimum(nodeEnd[i],maxTime+1);
	nodeReward[id]=nodeReward[i];
        maxReward+=nodeReward[id];
	id++;
      }

    std::cout << "Max Reward -> " << maxReward << std::endl;
    for (int ti=1; ti<=maxTime+1; ti++)
      nodeDepartProbs[0][ti]=0;
    nodeDepartProbs[0][0]=1;
    nodePCost[0]=0;
    currentArrivalDistribution=new double[maxTime+2];


    ifstream adjStream( _arcFile );
    numberOfLines=0;
    id=0;
    int realNode=-1;
    //int scenario=0;
    while( getline( adjStream, temp ) )
      {
	//cout << "WARNING: FIRST SCENARIO MULTIPLIED WITH 0.505!!\n";
	//if (id==0) cout << "\n\nScenario " << scenario << endl;
	numberOfLines++;
	vector<string> elems;
	splitString(temp, ',', elems);
	if (elems[0][0]=='#') continue;
	if (unreachable.isIn(++realNode)) continue;
	//std::cout << id << "=" << realNode << " ";
	nodeMapping[id]=realNode;
	if (elems.size()!=(unsigned)n)
	  {
	    cout << "ARC INPUT FILE ERROR: LINE " << numberOfLines << endl;
	    exit (999);
	  }	
	for (int hi=realNode,i=id; hi<n; hi++)
	  {
	    assert((double)atof(elems[i].c_str())>=0);
	    if (unreachable.isIn(hi)) continue;
	    maxArcTime[i][id]=maxArcTime[id][i]=(double)atof(elems[hi].c_str());
	    i++;
	  }
	id++;
      }
    adjStream.close();
    //std::cout << std::endl;
    if (realNode<n-1)
      {
	cout << "ARC INPUT FILE ERROR - INCORRECT NUMBER OF LINES!\n";
	exit (999);
      }
    origN=n;
    n-=unreachable.card;
    //std::cout << "Reachable Nodes -> " << n << std::endl;
    for (int scenario=0; scenario<k; scenario++)
      for (int id=0; id<n; id++)
	for (int i=id; i<n; i++)
	  {
	    if (scenario==0) 
	      {
		arcTime[scenario][id][i]=maxArcTime[id][i]*0.505;
		arcTime[k+1][id][i]=maxArcTime[id][i];
		arcTime[k+1][i][id]=maxArcTime[id][i];
		//cout << arcTime[scenario][id][i] << " ";
	      }
	    else 
	      {
		arcTime[scenario][id][i]=0.01*roundf((100*maxArcTime[id][i])*myRand(1,100)*0.01);
		//cout << arcTime[scenario][id][i] << " ";
	      }
	    arcTime[scenario][i][id]=arcTime[scenario][id][i];
	  }
    minorSelection=new SSet(n);
    minorNormalizer=new double[n];
    minorTimePenalty=new double[maxTime+10];
    initMinorTimePenalty();
    for (int i=0; i<n; i++)
      {
	minorNormalizer[i]=minorTimePenalty[minimum(nodeEnd[i],maxTime)];
      }
    setReducedScenarioNumber(0.1);

    tourS=new int[n];
    tourP=new int[n];
    int ass[n];
    for (int i=0; i<n; i++) 
      {
	tourS[i]=tourP[i]=ass[i]=i;
	//ass[i]=i; //(i+1)%n;
      }
    tourSEnd=tourPEnd=0; //n-1;
    eval(ass);
    evalP(ass);
    simulateScen(0,0);
    simulateProb(0,0);
    //analyze();
  }



  ~TSPTW(void) 
  {
    for (int s=0; s<k+2; s++)
      {
	for (int i=0; i<origN; i++)
	  delete [] arcTime[s][i];
	delete [] arcTime[s];
	delete [] nodeSCost[s];
	delete [] nodeSTime[s];
      }
    for (int i=0; i<n; i++)
      delete [] nodeDepartProbs[i];
    delete [] nodeDepartProbs;
    delete [] nodePCost;
    delete minorSelection;
    delete [] nodePMinorObj;
    delete [] nodeSMinorObj;
    delete [] minorSNorm;
    delete [] minorPNorm;
    delete [] minorNormalizer;
    delete [] minorTimePenalty;
    delete [] arcTime;
    delete [] nodeSCost;
    delete [] nodeSTime;
    delete [] tourS;
    delete [] tourP;
    delete [] nodeStart;
    delete [] nodeEnd;
    delete [] nodeReward;
    delete [] selection;
    delete [] currentArrivalDistribution;
    delete [] nodeMapping;
  };


  double fullEval(int* ass)
  {
    double ret=evalP(ass);
    /*
    std::cout << ret << " <> "
	      << simulateProb(0,0) << std::endl;
    */
    assert(fabs(ret-simulateProb(0,0))<EPSILON);
    return ret;
  }

  double eval(int* ass)
  {
    bool eq=true;
    int mini=-1;
    tourSEnd=-1;
    for (int i=0; i<n; i++) 
      {
	if (tourS[i]!=ass[i])
	  {
	    if (eq) mini=i;
	    eq=false;
	    tourS[i]=ass[i];
	  }
	if (tourS[i]==0&&tourSEnd<0) 
	  {
	    tourSEnd=i;
	  }
      }
    assert(tourSEnd>=0);
    assert(tourSEnd<n);
    if (mini<0) //return tourSCost;
      {
	if (modifiedObjective)
	  return tourSCost-minorCoeff*tourSMinorObj;
	else
	  return tourSCost;
      }
    assert(mini<n);
    assert(mini>=0);
    if (mini==0) return simulateScen(0,0);     
    assert(mini>0);
    int oldNode=tourS[mini-1];
    return simulateScen(oldNode,mini);    
  }


  double evalP(int* ass)
  {
    bool eq=true;
    int mini=-1;
    tourPEnd=-1;
    for (int i=0; i<n; i++) 
      {
	assert(tourP[i]>=0 && tourP[i]<n);
	assert(ass[i]>=0 && ass[i]<n);
	if (tourP[i]!=ass[i])
	  {
	    if (eq) mini=i;
	    eq=false;
	    tourP[i]=ass[i];
	  }
	if (tourP[i]==0&&tourPEnd<0) 
	  {
	    tourPEnd=i;
	  }
      }
    assert(tourPEnd>=0);
    assert(tourPEnd<n);
    if (mini<0) 
      {
	/*
	std::cout << "evalP Mini = " << mini << std::endl
		  << "tourPCost = " << tourPCost 
		  << "  minorCoeff = " << minorCoeff 
		  << "  tourPMinorObj = " << tourPMinorObj << std::endl;
	*/
	if (modifiedObjective)
	  return tourPCost-minorCoeff*tourPMinorObj;
	else
	  return tourPCost;
      }
    assert(mini<n);
    assert(mini>=0);
    //std::cout << "evalP Mini = " << mini <<std::endl;
    if (mini==0) return simulateProb(0,0);     
    assert(mini>0);
    int oldNode=tourP[mini-1];
    return simulateProb(oldNode,mini);    
  }


  double eval(int a, int b, int x, int y)
  {
    /*
    cout << "\n\nEVAL SWAP -> " << a << "|" << b << "  " << x << "|" << y << "\n";
    printCurrent();
    */
    swap(tourS,tourSEnd,x,y);
    swap(tourS,tourSEnd,a,b);
    int mini=a;
    if (b<mini) mini=b;
    if (x<mini) mini=x;
    if (y<mini) mini=y;
    if (mini>tourSEnd) return tourSCost;
    if (mini==0) return simulateScen(0,0);
    int oldNode=tourS[mini-1];
    return simulateScen(oldNode,mini);
  }

  double evalP(int a, int b, int x, int y)
  {
    /*
    cout << "\n\nEVAL SWAP -> " << a << "|" << b << "  " << x << "|" << y << "\n";
    printCurrent();
    */
    swap(tourP,tourPEnd,x,y);
    swap(tourP,tourPEnd,a,b);
    int mini=a;
    if (b<mini) mini=b;
    if (x<mini) mini=x;
    if (y<mini) mini=y;
    if (mini>tourPEnd) return tourPCost;
    if (mini==0) return simulateProb(0,0);
    int oldNode=tourP[mini-1];
    return simulateProb(oldNode,mini);
  }

};





class LP: public LearnParam
{
  int nf;
  double* params;
  double** features;
 public:
  LP(int numFeatures, double* parameters): nf(numFeatures)
  {
    params = new double[nf+1];
    features = new double*[nf];
    for (int i=0; i<=nf; i++) params[i]=parameters[i];
  }
  ~LP(void)
  {
    delete [] params;
    delete [] features;
  }
  void setFeatures(double** references)
  {
    for (int i=0; i<nf; i++) features[i]=references[i];
  }
  double getPercentage(void)
  {
    double lin=params[nf];
    for (int i=nf-1; i>=0; i--) lin+=*(features[i])*params[i];
    double res=1/(1+exp(lin));
    return maximum(0,minimum(100,res*100+0.5));
  };
};


void usage(char* pname)
{
  cout << pname << " numberOfNodes numberOfScenarios nodefile arcfile nodePenalty solTimeLimit tourPenalty runTimeLimit[in ms] polyExp[%] infile outFile infile outFile scenarioSeed solSeed [seedfile type 0\1\2] [11*12 + 11 Hyperparameters]\n";
  exit (999);
}



//#define PARAM

int main(int argc, char** argv)
{
  int numRegParams=17;
  int numLearningParameters=11;
  int numSubParameters=12;

  int numParameters=numLearningParameters*numSubParameters;
  int totalParameters=numRegParams+numParameters;

#ifdef PARAM
  if (argc!=totalParameters+numSubParameters-1) usage(argv[0]);
#else
  if (argc!=numRegParams) usage(argv[0]);
#endif

  int n=atoi(argv[1]);
  int k=atoi(argv[2]);
  double nodePen=atof(argv[5]);
  int solTimeLimit=atoi(argv[6]);
  double tourPen=atof(argv[7]);
  int timeLimit=atoi(argv[8]);
  double polyExp=0.01*atoi(argv[9]);

  mySRand(atoi(argv[14]));
#ifdef MINOROBJ
  TSPTW tspminor(n,k,argv[3],argv[4],nodePen,solTimeLimit,tourPen,polyExp,true);
#endif
  TSPTW tsp(n,k,argv[3],argv[4],nodePen,solTimeLimit,tourPen,polyExp,false);
  double* params=new double[numParameters];
  mySRand(atoi(argv[15]));
  int startSeeds=atoi(argv[16]);

#ifdef PARAM
  for (int i=0; i<numParameters; i++)
    params[i]=atof(argv[i+numRegParams]);

  //cout << "Parameters\n";
  for (int k=0; k<numLearningParameters; k++)
    {
      for (int l=0; l<numSubParameters-1; l++)
	{
	  params[k*numSubParameters+l]*=atoi(argv[totalParameters+l])*0.1;
	  //	  std::cout << params[k*numSubParameters+l] << " ";
	}
      double h=params[(k+1)*numSubParameters-1]*0.001;
      if (h<EPSILON) params[(k+1)*numSubParameters-1]=100;
      else if (1-h<EPSILON) params[(k+1)*numSubParameters-1]=-100;
      else params[(k+1)*numSubParameters-1] = log(1./h - 1);
      
      //std::cout << params[(k+1)*numSubParameters-1] << "\n";
    }
#else
  
  double humanP[] = {0.2, //GreedyVars
                     0.7, //AntiGreedyProb
                     0, //AntithesisSizeMin
                     0.4, //AntithesisSizeMax
                     0.1, //AlternativeAntiSizeMin
                     0.4, //AlternativeAntiSizeMax 
                     0.5, //AlternativeAntiProb
                     0.5, //AlternativeAntiRemoveFirstProb
                     0.0085, //RestartProb
                     0, //RestartMinSize 
                     0.35,  //RestartMaxSize
                     };
  for (int i=0; i<numParameters; i++) params[i]=0;   
  for (int i=0; i<numLearningParameters; i++)
    {
      double val;
      if (humanP[i]<EPSILON) val=5000;
      else if (1-humanP[i]<EPSILON) val=-5000;
      else val = log(1./humanP[i] - 1);
      params[(i+1)*numSubParameters-1]=val;
    }
#endif  

  Variable* v=new Variable[n];
  for (int i=0; i<n; i++)
    v[i]=Variable(perm,0,n-1);

  LP* gV=new LP(11,&params[0]);
  LP* aG=new LP(11,&params[12]);
  LP* aMin=new LP(11,&params[24]);
  LP* aMax=new LP(11,&params[36]);
  LP* alMin=new LP(11,&params[48]);
  LP* alMax=new LP(11,&params[60]);
  LP* alProb=new LP(11,&params[72]);
  LP* aRProb=new LP(11,&params[84]);
  LP* nI=new LP(11,&params[96]);
  LP* rMin=new LP(11,&params[108]);
  LP* rMax=new LP(11,&params[120]);
#ifdef MINOROBJ
  TSPTWHegel hegel(tsp.getNumberOfNodes(), v, &tspminor, &tsp, startSeeds, 0.04, 0.06, gV, aG, aMin, aMax, alMin, alMax, alProb, aRProb, nI, rMin, rMax);
#else
  TSPTWHegel hegel(tsp.getNumberOfNodes(), v, &tsp, &tsp, startSeeds, 0.04, 0.06, gV, aG, aMin, aMax, alMin, alMax, alProb, aRProb, nI, rMin, rMax);
#endif
  delete [] params;
  
  double res=hegel.minimize(timeLimit,argv[10],argv[11],argv[12],argv[13],-100000);
  if (res<-100000)
    {
      //cout << "s OPTIMUM FOUND\n" << std::flush;
      res=0;
    }
  else 
    {
      //cout << "s UNKNOWN\nres -> " << res << std::endl << std::flush; //"unsatisfiable\n";
      res=1;
    }
  delete aG;
  delete aMin;
  delete aMax;
  delete alMin;
  delete alMax;
  delete alProb;
  delete aRProb;
  delete nI;
  delete gV;
  delete rMin;
  delete rMax;
  return res;
}
