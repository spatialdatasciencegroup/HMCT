#pragma once
#include<iostream>
#include<fstream>
#include<algorithm>
#include<numeric>
#include<vector>
#include<string>
#include<chrono>
#include<ctime>
#include<cmath>
#include<limits>
#include<cstdio>
#include<queue>
#include <stack> 
#include <list>
#include<unordered_set> 
#include <iomanip>
using namespace std;

struct Node {
	double elevation;
	int nodeIndex; // original index from raster
	vector<int> parentsID;  ///Contour Tree parents node indices
	vector<int> childrenID;  //Contour Tree child

							 //contour tree construction
	vector<int> joinParentsID;
	int joinChildID;
	int splitParentID;
	vector<int> splitChildrenID;
	int upDegree; // for tree merge convinence
	int downDegree;

	//collapse tree
	vector<int> collapsedPixelIDs;

	//message propagation
	//leaves to root
	vector<double> fi_ChildList;
	double fi_parent[cNum];
	double fo[cNum];
	int foNode;
	int foNode_ischild;

	//root to leaves
	vector<double> go_ChildList;
	double go_parent[cNum];
	double gi[cNum];
	bool go_fromParent;
	bool go_fromChild;
	bool go_fromParentofChild;
	int go_lastVisitedNode;

	//inference
	//added by Arpan
	vector<int> correspondingNeighbour;
	vector<int> correspondingNeighbourClassOne;
	vector<int> correspondingNeighbourClassZero;
	Node() {
	}
	Node(double value, int nodeindex) {
		elevation = value;
		nodeIndex = nodeindex;
		splitParentID = -1;
		joinChildID = -1;

	}
};



struct sParameter {
	double Mu[cNum][Dim] = { 0.0 }; // True value, Mu, Sigma are related to p(yi=0|X,theta) , p(yi=1|X,theta)
	double elnMu[cNum][Dim] = { 0.0 }; // eln value
	double Sigma[cNum][Dim][Dim] = { 1.0 };

	double Epsilon = 0.01; // Epsilon: p(zi=0|zpi=1) = Epsilon, is related to p(zi=0, zpi=1|X,theta}, p(zi=1, zpi=1|X,theta) when i has parents
	double Pi = 0.5; // Pi: p(zi = 1) = Pi, is related to p(zi=0|X,theta}, p(zi =1|X,theta) when i has no parents
					 //double M = 0.4; // M: M = y0GivenZ1, is related to p(yi=0,zi=1|X,theta), p(yi=1, zi=1|X,theta)
	int CollapseSwitch = 0;

	vector<double> elnPxn_zn;
	double elnPy_z[cNum][cNum]; //transition probabilities
	double elnPz[cNum];
	double elnPz_zpn[cNum][cNum];

	int allPixelSize = -1; //set a wrong default value, so it will report error if it's not initialized properly
	double THRESHOLD = 0.05;
	int maxIteratTimes = 30;
	int maxParentDegree = 10;
	int ROW = -1;
	int COLUMN = -1;
	int NAValue = -1;
	int orgPixelSize = -1;
	int bfsRoot = -1;
};


struct sData {
	vector<bool>NA;
	vector<float>features; // RGB +..., rowwise, long array
	vector<double>elevationVector;
	vector<int> sortedElevationIndex;
	vector< pair<double, int> >elevationIndexPair; // here index means expanded index
	vector<Node*>allNodes;
	vector<double>A;
	vector<double>B;
	vector<int>C;
	vector<int> bfsTraversalOrder;
};


struct sTree {
	queue<int> leavesIDQueue;
	vector<int>nodeLevels;
};


struct sInference {
	vector<double> incoming;
	vector<double> outgoing;

	vector<double>waterLikelihood;  //-0.5(x-mu)T Sigma^-1 (x-mu) for class 1
	vector<double>dryLikelihood;  //-0.5(x-mu)T Sigma^-1 (x-mu) for class 0
	double lnCoefficient[cNum]; //log constanct of Gaussian distribution for two classes

	vector<double> lnTop2y;
	vector<double> lnz2top; //from z to y, outgoing of z, before factor node
	vector<double> lnbottom2y; //Vertical incoming message from z to y, after factor node

	vector<double> lnvi; // Top2z message
	vector<double> lnfi; // leaves to root, incoming messages; from zp to z except zp=NULL, after factor node
	vector<double> lnfo; // leaves to root, outgoing messages; product of lnfi and lngi, before factor node
	vector<double> lngi; // root to leaves, incoming messages; Horizontal incoming message from z to zp, after factor node
	vector<double> lngo; // root to leaves, outgoing messages

	vector<double> marginal_Yn;
	vector<double> marginal_YnZn; // only zn = 1 is useful to update parameters
	vector<double> marginal_ZnZpn; // only Zpn = 1 is useful to update parameters
	vector<double> marginal_Zn; // Separate from ZnZpn now. Those internal values are just for testing.

	vector<int>chainMaxGainFrontierIndex;

	//added by Arpan
	vector<vector<int>> correspondingNeighbours;
	vector<vector<int>> correspondingNeighboursClass;
};

struct subset
{
	int parent;
	int rank;
};


struct conMatrix {
	int TT;
	int TF;
	int FF;
	int FT;
};