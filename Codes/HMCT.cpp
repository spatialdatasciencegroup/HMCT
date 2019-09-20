//Assumption no missing Value in this test data
#define Dim 3 //input data dimension
#define cNum 2
#define _USE_MATH_DEFINES
#define LOGZERO -INFINITY
#define _CRT_SECURE_NO_WARNINGS
#define MESSAGELOW -INFINITY
#define MESSAGEHIGH 10000
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

#include "DataTypes.h"
using namespace std;


class cFlood {
private:
	//struct sComponent unionComponent;
	struct sParameter parameter;
	struct sData data;
	struct sTree tree;
	struct sInference infer;
	vector<int>testIndex;
	vector<int>testLabel;
	vector<int>mappredictions;
	//ofstream timeLogger;
	std::string CTInputLocation;
	std::string CTDem;
	std::string CTFeature;
	std::string CTPara;
	std::string CTTrain;
	std::string CTOutputLocation;
	std::string CTPrediction;
	//std::string CTParaLog;
	//std::string CTTimeLog;
	std::string CTTestIndex;
	std::string CTTestLabel;

	//tree construction
	struct subset *subsets;

public:
	void input(int argc, char *argv[]);
	void joinTree();
	void splitTree();
	void mergeTree();
	void treeConstrut();
	void MessagePropagation();
	void learning();
	void Collapselearning();

	void UpdateTransProb(); //Update P(y|z), P(zn|zpn), P(zn|zpn=empty)
	void UpdatePX_Z(); //Update P(x|y) based on parameters
	void CollapseUpdatePX_Z();
	void UpdateParameters();
	void CollapseUpdateParameters();
	void UpdateMarginalProb();

	//utilities
	int find(struct subset subsets[], int i);
	void Union(struct subset subsets[], int x, int y);

	//collapse tree
	void collapse();

	//inference 
	void inference();
	void output();

	//helper functions
	void computeMuandSigma();
	void removeLink(vector<int>&v, int removeID);
	void displayTree(int TreeID);
	void collapseWithMaxDegree();
	void updateMapPrediction();
	//struct conMatrix getConfusionMatrix();
};



void getCofactor(double mat[Dim][Dim], double temp[Dim][Dim], int p, int q, int n) {
	int i = 0, j = 0;
	// Looping for each element of the matrix
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			//  Copying into temporary matrix only those element
			//  which are not in given row and column
			if (row != p && col != q) {
				temp[i][j++] = mat[row][col];

				// Row is filled, so increase row index and
				// reset col index
				if (j == n - 1) {
					j = 0;
					i++;
				}
			}
		}
	}
}

//dynamic memory allocation,dimensional two dimension array
/* Recursive function for finding determinant of matrix.
n is current dimension of mat[][]. */
double determinant(double mat[Dim][Dim], int n) {
	double D = 0; // Initialize result

				  //  Base case : if matrix contains single element
	if (n == 1)
		return mat[0][0];

	double temp[Dim][Dim]; // To store cofactors
	int sign = 1;  // To store sign multiplier

				   // Iterate for each element of first row
	for (int f = 0; f < n; f++) {
		// Getting Cofactor of mat[0][f]
		getCofactor(mat, temp, 0, f, n);
		D += sign * mat[0][f] * determinant(temp, n - 1);

		// terms are to be added with alternate sign
		sign = -sign;
	}
	return D;
}

// Function to get adjoint of A[Dim][Dim] in adj[Dim][Dim].
void adjoint(double A[Dim][Dim], double adj[Dim][Dim]) {
	if (Dim == 1) {
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][]
	int sign = 1;
	double temp[Dim][Dim];

	for (int i = 0; i < Dim; i++) {
		for (int j = 0; j < Dim; j++) {
			// Get cofactor of A[i][j]
			getCofactor(A, temp, i, j, Dim);

			// sign of adj[j][i] positive if sum of row
			// and column indexes is even.
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the
			// transpose of the cofactor matrix
			adj[j][i] = (sign)*(determinant(temp, Dim - 1));
		}
	}
}

// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(double A[Dim][Dim], double inverse[Dim][Dim]) {
	// Find determinant of A[][]

	if (Dim == 1) {
		inverse[0][0] = 1.0 / A[0][0];
		return true;
	}

	double det = determinant(A, Dim);
	if (det == 0) {
		cout << "Singular matrix, can't find its inverse";
		return false;
	}

	// Find adjoint
	double adj[Dim][Dim];
	adjoint(A, adj);

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)"
	for (int i = 0; i < Dim; i++)
		for (int j = 0; j < Dim; j++)
			inverse[i][j] = adj[i][j] / double(det);
	return true;
}

// extended ln functions
double eexp(double x) {
	if (x == LOGZERO) {
		return 0;
	}
	else {
		return exp(x);
	}
}

double eln(double x) {
	if (x == 0) {
		return LOGZERO;
	}
	else if (x > 0) {
		return log(x);
	}
	else {
		cout << "Negative input error " << x << endl;
		exit(0);
	}
}

double elnsum(double x, double y) {
	if (x == LOGZERO) {
		return y;
	}
	else if (y == LOGZERO) {
		return x;
	}
	else if (x > y) {
		return x + eln(1 + eexp(y - x));
	}
	else {
		return y + eln(1 + eexp(x - y));
	}
}

double elnproduct(double x, double y) {
	if (x == LOGZERO || y == LOGZERO) {
		return LOGZERO;
	}
	else {
		return x + y;
	}
}

void cFlood::UpdateTransProb() {
	if (cNum != 2) {
		cout << "cannot handle more than two classes now!" << endl;
		std::exit(1);
	}

	double eln(double);
	parameter.elnPz[0] = eln(1 - eexp(parameter.Pi));
	parameter.elnPz[1] = parameter.Pi;
	parameter.elnPz_zpn[0][0] = eln(1);
	parameter.elnPz_zpn[0][1] = parameter.Epsilon;
	parameter.elnPz_zpn[1][0] = eln(0);
	parameter.elnPz_zpn[1][1] = eln(1 - eexp(parameter.Epsilon));
	if (eexp(parameter.Epsilon) < 0 || eexp(parameter.Epsilon) > 1) {
		cout << "Epsilon Error: " << eexp(parameter.Epsilon) << endl;
	}
	if (eexp(parameter.Pi) < 0 || eexp(parameter.Pi) > 1) {
		cout << "Pi Error: " << eexp(parameter.Pi) << endl;
	}
	if (eexp(parameter.elnPz_zpn[0][1]) + eexp(parameter.elnPz_zpn[1][1]) != 1) {
		cout << "Error computing parameter.elnPz_zpn " << endl;
	}
	if (eexp(parameter.elnPz[0]) + eexp(parameter.elnPz[1]) != 1) {
		cout << "Error computing parameter.elnPz " << endl;
	}
}

void cFlood::UpdatePX_Z() {
	// Calculate inverse of sigma
	double adjointMatrix[cNum][Dim][Dim]; // To store adjoint of A[][]
	double inverseMatrix[cNum][Dim][Dim]; // To store inverse of A[][]
	for (int c = 0; c < cNum; c++) {
		adjoint(parameter.Sigma[c], adjointMatrix[c]);
	}
	for (int c = 0; c < cNum; c++) {
		if (!inverse(parameter.Sigma[c], inverseMatrix[c])) {
			cout << "Inverse error" << endl;
		}
	}

	//xiGivenZi_coefficient, log form
	for (int c = 0; c < cNum; c++) {// |Sigma|^(-1/2) 
		infer.lnCoefficient[c] = -0.5 * Dim * log(2 * M_PI) - 0.5 * log(fabs(determinant(parameter.Sigma[c], Dim)));
	}

	// Calculate p(x|z)
	double intermediateValue[cNum][Dim] = { 0 };
	double likelihood[cNum] = { 0 };
	double xMinusMu[cNum][Dim] = { 0 };

	for (size_t i = 0; i < parameter.allPixelSize; i++) {
		if (!data.NA[i]) { // Not missing data

			for (int c = 0; c < cNum; c++) {
				likelihood[c] = 0;
			}

			for (int c = 0; c < cNum; c++) {
				for (int d = 0; d < Dim; d++) {
					intermediateValue[c][d] = 0;
				}
			}

			// -0.5*(x-mu)' * Sigma^-1 * (x-mu), matrix multiply
			for (int c = 0; c < cNum; c++) {
				for (int d = 0; d < Dim; d++) {
					xMinusMu[c][d] = data.features[i * Dim + d] - parameter.Mu[c][d];
				}
			}

			for (int c = 0; c < cNum; c++) {
				for (int k = 0; k < Dim; k++) {
					for (int n = 0; n < Dim; n++) {
						intermediateValue[c][k] += xMinusMu[c][n] * inverseMatrix[c][n][k];
					}
					likelihood[c] += intermediateValue[c][k] * xMinusMu[c][k];
				}
			}

			for (int cls = 0; cls < cNum; cls++) {
				parameter.elnPxn_zn[i*cNum + cls] = -0.5 * likelihood[cls] + infer.lnCoefficient[cls];
			}

		}
		else {
			for (int cls = 0; cls < cNum; cls++) {
				parameter.elnPxn_zn[i*cNum + cls] = eln(1);
			}
		}

	}
}


void cFlood::CollapseUpdatePX_Z() {
	// Calculate inverse of sigma
	double adjointMatrix[cNum][Dim][Dim]; // To store adjoint of A[][]
	double inverseMatrix[cNum][Dim][Dim]; // To store inverse of A[][]
	double MuMuT[cNum][Dim][Dim];

	for (int c = 0; c < cNum; c++) {
		adjoint(parameter.Sigma[c], adjointMatrix[c]);
	}
	for (int c = 0; c < cNum; c++) {
		if (!inverse(parameter.Sigma[c], inverseMatrix[c])) {
			cout << "Inverse error" << endl;
		}
	}

	//xiGivenZi_coefficient, log form
	for (int c = 0; c < cNum; c++) {// |Sigma|^(-1/2) 
		infer.lnCoefficient[c] = -0.5 * Dim * log(2 * M_PI) - 0.5 * log(fabs(determinant(parameter.Sigma[c], Dim)));
	}

	// Calculate p(x|z)
	for (int c = 0; c < cNum; c++) {
		for (size_t m = 0; m < Dim; m++) { // row
			for (size_t n = 0; n < Dim; n++) { // column
				MuMuT[c][m][n] = parameter.Mu[c][m] * parameter.Mu[c][n];
			}
		}
	}

	double intermediateValue[cNum][Dim][Dim] = { 0 };
	double likelihood[cNum] = { 0 };

	for (size_t i = 0; i < parameter.allPixelSize; i++) {
		if (!data.NA[i]) { // Not missing data
												//compute intermediate value array for node i
			for (int c = 0; c < cNum; c++) {
				likelihood[c] = 0;
			}

			for (int c = 0; c < cNum; c++) {

				for (size_t m = 0; m < Dim; m++) { // row
					for (size_t n = 0; n < Dim; n++) { // column
						intermediateValue[c][m][n] = data.A[i * Dim * Dim + m * Dim + n] - data.B[i * Dim + m] * parameter.Mu[c][n]
							- data.B[i * Dim + n] * parameter.Mu[c][m] + data.C[i] * MuMuT[c][m][n];
					}
				}

				//dot product on SigmaInverse[c] with intermediateValue array
				for (size_t m = 0; m < Dim; m++) { // row
					for (size_t n = 0; n < Dim; n++) { // column
						likelihood[c] += inverseMatrix[c][m][n] * intermediateValue[c][m][n];
					}
				}

			}


			for (int cls = 0; cls < cNum; cls++) {
				parameter.elnPxn_zn[i*cNum + cls] = -0.5 * likelihood[cls] + data.C[i] * infer.lnCoefficient[cls];
			}

			//verifying
			for (int cls = 0; cls < cNum; cls++) {
				if (parameter.elnPxn_zn[i*cNum + cls] < MESSAGELOW || parameter.elnPxn_zn[i*cNum + cls] > MESSAGEHIGH) {
					cout << "Error computing parameter.elnPxn_zn for Node " << i << endl;
				}
			}

		}
		//else {// misssing data //p(x|z) will be assigned a constant directly, no need to calculate likelihood
		//	infer.dryLikelihood[i] = eln(0.5);
		//	infer.waterLikelihood[i] = eln(0.5);
		//}
	}
}

vector<int> getBFSOrder(int root, vector<Node*> &allNodes, int allPixelSize) {
	vector<int> bfsVisited;
	vector<int> bfs;
	bfsVisited.resize(allPixelSize, 0);
	queue<int> que;
	que.push(root);

	while (!que.empty()) {
		int currentNode = que.front();
		bfs.push_back(currentNode);
		bfsVisited[currentNode] = 1;
		que.pop();
		for (int i = 0; i < allNodes[currentNode]->childrenID.size(); i++) {
			int child = allNodes[currentNode]->childrenID[i];
			if (!bfsVisited[child]) {
				que.push(child);
			}
		}
		for (int i = 0; i < allNodes[currentNode]->parentsID.size(); i++) {
			int parent = allNodes[currentNode]->parentsID[i];
			if (!bfsVisited[parent]) {
				que.push(parent);
			}
		}
	}
	return bfs;
}

void cFlood::computeMuandSigma() {
	ifstream trainFile(CTInputLocation + CTTrain);
	if (!trainFile) {
		std::cout << "Missing Training Data.." << std::endl;
		exit(0);
	}
	vector<int> trainData;
	int traindata;
	while (trainFile >> traindata)
	{
		trainData.push_back(traindata); 
	}
	double tempSigma[cNum][Dim][Dim] = { 0 };
	int samplesPerCass[cNum] = { 0 };
	int trainCount = trainData.size() / (Dim + 1);
	for (int cls = 0; cls < cNum; cls++) {
		for (int i = 0; i < trainCount; i++) {
			if (trainData[i*(Dim + 1) + Dim] == cls) {
				samplesPerCass[cls]++;
				for (int col = 0; col < Dim; col++) {
					parameter.Mu[cls][col] += trainData[i*(Dim + 1) + col];
				}
			}
		}

		for (int col = 0; col < Dim; col++) {
			parameter.Mu[cls][col] /= samplesPerCass[cls];
		}
	}

	double xMinusMu[cNum][Dim];
	for (size_t i = 0; i < trainCount; i++) {

		for (int c = 0; c < cNum; c++) {
			for (size_t j = 0; j < Dim; j++) {
				if (trainData[i * (Dim + 1) + Dim] == c)
					xMinusMu[c][j] = trainData[i * (Dim + 1) + j] - parameter.Mu[c][j];
			}
		}

		for (int c = 0; c < cNum; c++) {
			for (size_t m = 0; m < Dim; m++) { // row
				for (size_t n = 0; n < Dim; n++) { // column
					if (trainData[i * (Dim + 1) + Dim] == c)
						parameter.Sigma[c][m][n] += xMinusMu[c][m] * xMinusMu[c][n];
				}
			}
		}
	}
	for (int c = 0; c < cNum; c++) {
		for (size_t m = 0; m < Dim; m++) { // row
			for (size_t n = 0; n < Dim; n++) { // column
				parameter.Sigma[c][m][n] /= samplesPerCass[c];
			}
		}
	}

}

void cFlood::input(int argc, char *argv[]) {
	clock_t start_s = clock();
	if (argc > 1) {
		ifstream config(argv[1]);
		string line;
		getline(config, line);
		CTInputLocation = line;  //Input file location 
		getline(config, line);
		CTDem = line;           //Elevation data file name
		getline(config, line);
		CTFeature = line;       //Feature data file name 
		//getline(config, line);
		//CTIndex = line;         //Index data file name
		getline(config, line);
		CTPara = line;          //parameter data file name
		getline(config, line);
		CTTrain = line;     //Train data file name
		//getline(config, line);
		//CTTestIndex = line;     //test index data file name
		//getline(config, line);
		//CTTestLabel = line;    //file name of file containing ground truth information for each test index

		getline(config, line);
		CTOutputLocation = line; //oputput location to store the output of HMCT
		getline(config, line);
		CTPrediction = line;    //file name for output prediction data
		//getline(config, line);
		//CTParaLog = line;       //parameter log file name
		//getline(config, line);
		//if (line != "") {
		//	CTTimeLog = line;  //time log file name
		//}
		//else {
		//	std::cout << "config file error";
		//}
	}
	else {
		std::cout << "Missing Configuration File!";
	}
	//std::string timelogFile = CTOutputLocation + CTTimeLog;
	//timeLogger.open(timelogFile.c_str(), std::ofstream::app);

	ifstream elevationFile(CTInputLocation + CTDem);
	ifstream featuresFile(CTInputLocation + CTFeature);
	ifstream parameterFile(CTInputLocation + CTPara);
	//ifstream testIndexFile(CTInputLocation + CTTestIndex);
	//ifstream testLabelFile(CTInputLocation + CTTestLabel);


	if (!parameterFile) {
		std::cout << "Failed to open parameter!" << endl;
		exit(0);
	}

	if (!elevationFile ) {
		std::cout << "Failed to Elevation file" << endl;
		exit(0);
	}
	if (!featuresFile) {
		std::cout << "Filed to open Feature file" << endl;
		exit(0);
	}
	//if ( !testIndexFile ) {
	//	std::cout << "Failed to open Test Index file" << endl;
	//	exit(0);
	//}
	//if (!testIndexFile || !testLabelFile) {
	//	std::cout << "Failed to open Test Label file" << endl;
	//	exit(0);
	//}

	parameterFile >> parameter.THRESHOLD;
	parameterFile >> parameter.maxIteratTimes;
	parameterFile >> parameter.Epsilon;
	parameterFile >> parameter.Pi;
	parameterFile >> parameter.CollapseSwitch;
	parameterFile >> parameter.maxParentDegree; //this will be unused if collapseSwitch is off. 
	parameterFile >> parameter.ROW;
	parameterFile >> parameter.COLUMN;
	parameterFile >> parameter.NAValue;
 
	computeMuandSigma(); //get initial Mu and Sigma from training data 

	//int testIdx;
	//while (testIndexFile >> testIdx)
	//{
	//	testIndex.push_back(testIdx - 1); //testIdx extracted by R, starting from 1
	//}
	//int testLbl;
	//while (testLabelFile >> testLbl)
	//{
	//	testLabel.push_back(testLbl);
	//}

	if (parameter.Epsilon > 1 || parameter.Pi > 1) {
		cout << "wrong parameter" << endl;
	}

	cout << "Input parameters:" << endl << "Epsilon: " << parameter.Epsilon << " Pi: " << parameter.Pi; 

	parameter.allPixelSize = parameter.ROW * parameter.COLUMN;
	parameter.orgPixelSize = parameter.allPixelSize;

	data.features.resize(parameter.allPixelSize * Dim);// RGB + ..., rowwise, long array

	for (size_t i = 0; i < parameter.allPixelSize * Dim; i++) {
		featuresFile >> data.features[i];
	}

	data.NA.resize(parameter.allPixelSize);
	std::fill(data.NA.begin(), data.NA.end(), false);
	for (size_t i = 0; i < parameter.allPixelSize; i++) {
		for (int j = 0; j < Dim; j++) {
			if (data.features[i*Dim + j] == parameter.NAValue) {
				data.NA[i] = true;
				break;
			}
		}
	}

	data.elevationVector.resize(parameter.allPixelSize);
	for (size_t i = 0; i < parameter.allPixelSize; i++) {
		elevationFile >> data.elevationVector[i];
	}

	data.allNodes.resize(parameter.allPixelSize);
	for (int i = 0; i < parameter.allPixelSize; i++) { // only nonempty nodes will be generated
		data.allNodes[i] = new Node(data.elevationVector[i], i);
	}

	clock_t stop_s = clock();
	std::cout << "Data prepare " << endl << "time: " << (stop_s - start_s) / float(CLOCKS_PER_SEC) << endl << endl;
	//timeLogger << parameter.allPixelSize << "," << parameter.maxIteratTimes << "," << parameter.maxParentDegree << ",";
	// Contour Tree Construction Started
	cout << "Contour Tree Construction Started.." << endl << endl;
	auto start = std::chrono::system_clock::now();
	data.elevationIndexPair.resize(parameter.allPixelSize);
	data.sortedElevationIndex.resize(parameter.allPixelSize);
	for (size_t i = 0; i < parameter.allPixelSize; i++) {
		data.elevationIndexPair[i] = make_pair(data.elevationVector[i], i);
	}
	sort(std::begin(data.elevationIndexPair), std::end(data.elevationIndexPair));

	for (int i = 0; i < parameter.allPixelSize; i++) {
		int index = data.elevationIndexPair[i].second;
		data.sortedElevationIndex[index] = i;
	}

	joinTree();
	splitTree();
	mergeTree();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	//timeLogger << elapsed_seconds.count() << ",";  //recording CT contruction time
	cout << "Collapse Contour Tree Construction Finished. Duration:" << elapsed_seconds.count() << endl;
	// Contour Tree Construction Finished

	//Collapse Contour Tree Construction Started

	if (parameter.CollapseSwitch) {
		cout << "Collapse Contour Tree Construction Started.." << endl;
		start = std::chrono::system_clock::now();
		//collapse();
		collapseWithMaxDegree();  //degree provide as parameter
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		//timeLogger << elapsed_seconds.count() << ","; //recording collapse tree construction time 
		cout << "Collapse Contour Tree Construction Finished. Duration: " << elapsed_seconds.count() << endl << endl;
	}
	//else {
	//	timeLogger << 0.0 << ","; //no collapse time for uncollapsed tree
	//}

	//Collapse Tree Construction Finished

	//Collapse Tree ABC computation Started

	if (parameter.CollapseSwitch) {
		cout << "Collapse Contour Tree Construction Started.." << endl;
		start = std::chrono::system_clock::now();
		//computing data(i).A , data[i].B , data[i].C
		data.C.resize(parameter.allPixelSize, 0);
		data.B.resize(parameter.allPixelSize*Dim, 0);
		data.A.resize(parameter.allPixelSize*Dim*Dim, 0);
		for (int i = 0; i < parameter.allPixelSize; i++) {
			//computing C
			data.C[i] = (int)data.allNodes[i]->collapsedPixelIDs.size();

			//Computing B
			for (int p = 0; p < data.allNodes[i]->collapsedPixelIDs.size(); p++) {
				int pixelId = data.allNodes[i]->collapsedPixelIDs[p];
				for (int r = 0; r < Dim; r++) {
					data.B[i*Dim + r] += data.features[pixelId*Dim + r];
				}
			}

			//Computing A
			for (int p = 0; p < data.allNodes[i]->collapsedPixelIDs.size(); p++) {
				int pixelId = data.allNodes[i]->collapsedPixelIDs[p];
				for (int r = 0; r < Dim; r++) {
					for (int c = 0; c < Dim; c++) {
						data.A[i*Dim*Dim + r * Dim + c] += data.features[pixelId*Dim + r] * data.features[pixelId*Dim + c];
					}
				}
			}
		}
		//Collapse Tree ABC computation Finished
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		//timeLogger << elapsed_seconds.count() << ",";  //recording ABC computation time 
		cout << "Contour Tree Construction Finished. Duration: " << elapsed_seconds.count() << endl << endl;
	}
	//else {
	//	timeLogger << 0.0 << ",";  //no ABC computation time for uncollapsed tree
	//}


	//Learing Start
	cout << "Learning Started.." << endl;
	start = std::chrono::system_clock::now();
	//get bfs order
	if (parameter.bfsRoot == -1) {
		for (int i = 0; i < parameter.allPixelSize; i++) {
			if (data.allNodes[i]->childrenID.size() == 0) {
				parameter.bfsRoot = i;
				break;
			}
		}
	}
	data.bfsTraversalOrder = getBFSOrder(parameter.bfsRoot, data.allNodes, parameter.allPixelSize);

	parameter.elnPxn_zn.resize(parameter.allPixelSize*cNum);

	double determinantValue[cNum];
	for (int c = 0; c < cNum; c++) {
		determinantValue[c] = determinant(parameter.Sigma[c], Dim);
	}
	for (int c = 0; c < cNum; c++) {
		infer.lnCoefficient[c] = -0.5 * Dim * log(2 * M_PI) - 0.5 * log(fabs(determinantValue[c])); // |Sigma|^(-1/2), xiGivenYi_coefficient0
	}

	//convert parameter Pi, M, Epsilon to log form
	parameter.Pi = eln(parameter.Pi);
	parameter.Epsilon = eln(parameter.Epsilon); //check if already eln form?

	//setting up g;
	if (parameter.CollapseSwitch) {
		this->CollapseUpdatePX_Z();
	}
	else {
		this->UpdatePX_Z();
	}

	this->UpdateTransProb();

	if (parameter.CollapseSwitch) {
		Collapselearning();
	}
	else {
		learning();
	}
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	//timeLogger << elapsed_seconds.count() << ",";  //recording learning cost
												   //Learning End
	cout << "Learning Finished. Duration: " << elapsed_seconds.count() << endl << endl;

	cout << "Inference Started.." << endl;
	start = std::chrono::system_clock::now();
	inference();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	//timeLogger << elapsed_seconds.count() << std::endl; //recording inference cost
	//timeLogger.close();
	cout << "Inference Finished. Duration: " << elapsed_seconds.count() << endl << endl;
	output();

}

int cFlood::find(struct subset subsets[], int i)
{
	// find root and make root as parent of i (path compression) 
	if (subsets[i].parent != i)
		subsets[i].parent = find(subsets, subsets[i].parent);

	return subsets[i].parent;
}

// A function that does union of two sets of x and y 
// (uses union by rank) 
void cFlood::Union(struct subset subsets[], int x, int y)
{
	int xroot = find(subsets, x);
	int yroot = find(subsets, y);

	// Attach smaller rank tree under root of high rank tree 
	// (Union by Rank) 
	if (subsets[xroot].rank < subsets[yroot].rank)
		subsets[xroot].parent = yroot;
	else if (subsets[xroot].rank > subsets[yroot].rank)
		subsets[yroot].parent = xroot;

	// If ranks are same, then make one as root and increment 
	// its rank by one 
	else
	{
		subsets[yroot].parent = xroot;
		subsets[xroot].rank++;
	}
}

void cFlood::joinTree() {
	int curIdx, neighborIndex;
	int row, column;
	vector<int> lowestVertex(parameter.allPixelSize);
	subsets = (struct subset*) malloc(parameter.allPixelSize * sizeof(struct subset));
	for (int v = 0; v < parameter.allPixelSize; ++v)
	{
		subsets[v].parent = v;
		subsets[v].rank = 0;
		lowestVertex[v] = v;
	}

	for (int i = parameter.allPixelSize - 1; i >= 0; i--) {
		curIdx = data.elevationIndexPair[i].second;
		row = curIdx / parameter.COLUMN;
		column = curIdx % parameter.COLUMN;
		int h1 = data.sortedElevationIndex[curIdx];
		// check all 8 neighbors
		for (int j = max(0, row - 1); j <= min(parameter.ROW - 1, row + 1); j++) {
			for (int k = max(0, column - 1); k <= min(parameter.COLUMN - 1, column + 1); k++) {
				neighborIndex = j * parameter.COLUMN + k; //25

				if (data.NA[neighborIndex] == false && neighborIndex != curIdx) { // skip NA neighbor 
					int h2 = data.sortedElevationIndex[neighborIndex];
					if (h1 < h2) {
						int neighComponentID = find(subsets, neighborIndex);
						int currentComponetID = find(subsets, curIdx);
						if (neighComponentID == currentComponetID) {  //this means same as root2 == root1 but we don't need to find root2  //idea if they have same room they will point to same lowest vertex   
							continue;
						}
						int currentLowestNodeIdx = lowestVertex[neighComponentID];
						Union(subsets, curIdx, neighborIndex);
						if (data.allNodes[currentLowestNodeIdx]->joinChildID != -1) {
							cout << "join problem" << endl;
							//cout << currentLowestNodeIdx << "->" << data.allNodes[currentLowestNodeIdx]->joinChildID<<"->"<<curIdx << endl;
						}
						data.allNodes[currentLowestNodeIdx]->joinChildID = curIdx;  //confirm 
						data.allNodes[curIdx]->joinParentsID.push_back(currentLowestNodeIdx);//child to parent in join and parent to child in split
						int newComponentID = find(subsets, neighborIndex);
						lowestVertex[newComponentID] = curIdx;
						lowestVertex[neighComponentID] = curIdx;

					}
				}
			}
		}// go throuth 8 neighbors
	}
	free(subsets);
	//displayTree(1);

}

void cFlood::splitTree() {
	int curIdx, neighborIndex;
	int row, column;
	vector<int> highestVertex(parameter.allPixelSize);
	subsets = (struct subset*) malloc(parameter.allPixelSize * sizeof(struct subset));
	for (int v = 0; v < parameter.allPixelSize; ++v)
	{
		subsets[v].parent = v;
		subsets[v].rank = 0;
		highestVertex[v] = v;
	}
	for (size_t i = 0; i < parameter.allPixelSize; i++) {
		curIdx = data.elevationIndexPair[i].second;
		row = curIdx / parameter.COLUMN;
		column = curIdx % parameter.COLUMN;
		highestVertex[curIdx] = curIdx;

		int h1 = data.sortedElevationIndex[curIdx];
		// check all 8 neighbors
		for (int j = max(0, row - 1); j <= min(parameter.ROW - 1, row + 1); j++) {
			for (int k = max(0, column - 1); k <= min(parameter.COLUMN - 1, column + 1); k++) {
				neighborIndex = j * parameter.COLUMN + k; //25

				if (data.NA[neighborIndex] == false && neighborIndex != curIdx) { // skip NA neighbor  
					int h2 = data.sortedElevationIndex[neighborIndex];
					if (h1 > h2) {
						int neighComponentID = find(subsets, neighborIndex);
						int currentComponetID = find(subsets, curIdx);
						if (neighComponentID == currentComponetID) {  //this means same as root2 == root1 but we don't need to find root2  //idea if they have same room they will point to same lowest vertex   
							continue;
						}
						int currentHighestNodeIdx = highestVertex[neighComponentID];
						Union(subsets, curIdx, neighborIndex);
						data.allNodes[currentHighestNodeIdx]->splitParentID = curIdx;
						data.allNodes[curIdx]->splitChildrenID.push_back(currentHighestNodeIdx);//child to parent in join and parent to child in split
						int newComponentID = find(subsets, curIdx);
						highestVertex[newComponentID] = curIdx;
					}
				}
			}
		}// go throuth 8 neighbors

	}
	//displayTree(2)
}


void cFlood::mergeTree() {
	int curIdx, totalDegree;
	for (int i = 0; i < parameter.allPixelSize; i++) {
		curIdx = data.elevationIndexPair[i].second;
		data.allNodes[curIdx]->upDegree = (int)data.allNodes[curIdx]->joinParentsID.size();
		data.allNodes[curIdx]->downDegree = (int)data.allNodes[curIdx]->splitChildrenID.size();

		totalDegree = data.allNodes[curIdx]->upDegree + data.allNodes[curIdx]->downDegree;
		if (data.allNodes[curIdx]->upDegree + data.allNodes[curIdx]->downDegree == 1) {
			tree.leavesIDQueue.push(curIdx);
		}
	}

	queue<int> myleaves = tree.leavesIDQueue;
	int iter = 0;
	while (!myleaves.empty()) {
		//traverse current node
		iter++;
		int curIdx = myleaves.front();
		myleaves.pop();
		int splitParent, joinParent, splitChild, joinChild;
		if (data.allNodes[curIdx]->joinParentsID.size() == 0 && data.allNodes[curIdx]->splitChildrenID.size() == 1) { //upper leaf 
																															//first copy the edge from the join tree to the contour tree 																												
			int newParent = data.allNodes[curIdx]->joinChildID;  // using the assumption node with lower elevation is parent
			data.allNodes[newParent]->childrenID.push_back(curIdx);
			data.allNodes[curIdx]->parentsID.push_back(newParent);
			//data.allNodes[curIdx]->adjNodes.push_back(newParent);

			//next remove the current node from the join tree (easy case)
			data.allNodes[curIdx]->joinChildID = -1; //no child
			data.allNodes[newParent]->joinParentsID.erase(std::find(data.allNodes[newParent]->joinParentsID.begin(), data.allNodes[newParent]->joinParentsID.end(), curIdx));
			data.allNodes[newParent]->upDegree = (int)data.allNodes[newParent]->joinParentsID.size();

			//next remove the current node from the split tree (need to mantain connectivity)
			splitParent = data.allNodes[curIdx]->splitParentID;
			if (data.allNodes[curIdx]->splitChildrenID.size() > 0) {
				splitChild = data.allNodes[curIdx]->splitChildrenID.back();
				//removing current node
				data.allNodes[curIdx]->splitChildrenID.pop_back(); //curIdx is a leaf and upDegree is already 0 so number of  splitChild = 1

			}
			else {
				splitChild = -1;
			}
			data.allNodes[curIdx]->splitParentID = -1;
			//fixing parent
			if (splitParent != -1) {
				data.allNodes[splitParent]->splitChildrenID.erase(std::find(data.allNodes[splitParent]->splitChildrenID.begin(), data.allNodes[splitParent]->splitChildrenID.end(), curIdx));
				data.allNodes[splitParent]->splitChildrenID.push_back(splitChild);
			}

			//fixing child
			if (splitChild != -1) {
				data.allNodes[splitChild]->splitParentID = splitParent;
			}
			//check if the child is a new leaf or not 
			if (data.allNodes[newParent]->joinParentsID.size() + data.allNodes[newParent]->splitChildrenID.size() == 1) {
				myleaves.push(newParent);
			}

		}
		else if (data.allNodes[curIdx]->joinParentsID.size() == 1 && data.allNodes[curIdx]->splitChildrenID.size() == 0) {  //lower leaf
																																  //first copy the edge from the split tree to the contour tree 
			int newChild = data.allNodes[curIdx]->splitParentID;  // using the assumption node with higher elevation is child
			data.allNodes[curIdx]->childrenID.push_back(newChild);
			data.allNodes[newChild]->parentsID.push_back(curIdx);
			//data.allNodes[curIdx]->adjNodes.push_back(newChild);

			//next remove the current node from the split tree (easy case)
			data.allNodes[curIdx]->splitParentID = -1;
			data.allNodes[newChild]->splitChildrenID.erase(std::find(data.allNodes[newChild]->splitChildrenID.begin(), data.allNodes[newChild]->splitChildrenID.end(), curIdx));
			data.allNodes[newChild]->downDegree = (int)data.allNodes[newChild]->splitChildrenID.size();

			//next remove the current node from the join tree (need to mantain connectivity)
			joinChild = data.allNodes[curIdx]->joinChildID;
			joinParent = data.allNodes[curIdx]->joinParentsID.back();
			//removing current node
			data.allNodes[curIdx]->joinParentsID.pop_back(); //curIdx is a leaf and downDegree is already 0 so number of  joinParent = 1
			data.allNodes[curIdx]->joinChildID = -1;
			//fixing parent
			if (joinParent != -1) {
				data.allNodes[joinParent]->joinChildID = joinChild;
			}
			//fixing child
			if (joinChild != -1) {
				data.allNodes[joinChild]->joinParentsID.erase(std::find(data.allNodes[joinChild]->joinParentsID.begin(), data.allNodes[joinChild]->joinParentsID.end(), curIdx));
				data.allNodes[joinChild]->joinParentsID.push_back(joinParent);
			}


			//check if the parent is a new leaf or not 
			if (data.allNodes[newChild]->joinParentsID.size() + data.allNodes[newChild]->splitChildrenID.size() == 1) {
				myleaves.push(newChild);
			}

		}
	}

	//Print contour Tree
	//displayTree(3);
}

void cFlood::displayTree(int TreeID) {
	if (TreeID == 1) {  //join Tree
		for (int i = 0; i < parameter.allPixelSize; i++) {
			if (data.allNodes[i]->joinParentsID.size() > 0) {
				cout << "<";
				for (int j = 0; j < data.allNodes[i]->joinParentsID.size(); j++) {
					if (j + 1 != data.allNodes[i]->joinParentsID.size())
						cout << data.allNodes[i]->joinParentsID[j] << ",";
					else
						cout << data.allNodes[i]->joinParentsID[j] << ">    <-----";
				}
			}
			else {
				cout << "<NUll>    <-----";
			}
			//cout << endl << "^"<<endl<<"|" << endl;

			cout << i << "------>     ";
			//cout  << "|" <<endl<<"v"<< endl;
			cout << data.allNodes[i]->joinChildID << endl << endl << endl;
		}
	}
	if (TreeID == 2) {  //split Tree
		for (int i = 0; i < parameter.allPixelSize; i++) {
			cout << data.allNodes[i]->splitParentID << "<----- ";
			cout << i << " ------>     ";
			if (data.allNodes[i]->splitChildrenID.size() > 0) {
				cout << "<";
				for (int j = 0; j < data.allNodes[i]->splitChildrenID.size(); j++) {
					if (j + 1 != data.allNodes[i]->splitChildrenID.size())
						cout << data.allNodes[i]->splitChildrenID[j] << ",";
					else
						cout << data.allNodes[i]->splitChildrenID[j] << ">" << endl << endl;
				}
			}
			else {
				cout << "<NUll>" << endl << endl;
			}
		}
	}
	else if (TreeID == 3) { //contour Tree
		cout << "contour Tree" << endl;
		for (int i = 0; i < parameter.allPixelSize; i++) {
			if (data.allNodes[i]->parentsID.size() > 0) {
				cout << "<";
				for (int j = 0; j < data.allNodes[i]->parentsID.size(); j++) {
					if (j + 1 != data.allNodes[i]->parentsID.size())
						cout << data.allNodes[i]->parentsID[j] << ",";
					else
						cout << data.allNodes[i]->parentsID[j] << ">    <-----";
				}
			}
			else {
				cout << "<NUll>    <-----";
			}
			cout << i << "------>     ";
			if (data.allNodes[i]->childrenID.size() > 0) {
				cout << "<";
				for (int j = 0; j < data.allNodes[i]->childrenID.size(); j++) {
					if (j + 1 != data.allNodes[i]->childrenID.size())
						cout << data.allNodes[i]->childrenID[j] << ",";
					else
						cout << data.allNodes[i]->childrenID[j] << ">" << endl << endl;
				}
			}
			else {
				cout << "<NUll>" << endl << endl;
			}

		}
	}

}

void cFlood::removeLink(vector<int> &v, int removeID) {
	v.erase(std::find(v.begin(), v.end(), removeID));
}


void cFlood::collapseWithMaxDegree() {
	int newNodeId = parameter.allPixelSize;
	vector<int> visited(parameter.allPixelSize * 2, 0);
	for (int tsIndex = 0; tsIndex < parameter.allPixelSize; tsIndex++) {
		int frontier = data.elevationIndexPair[tsIndex].second;/*topoSortedIndex[tsIndex];*/
		if (visited[frontier]) {
			continue;
		}
		double frontierHeight = data.allNodes[frontier]->elevation;
		data.allNodes.push_back(new Node(frontierHeight, newNodeId));  //maybe just push nodeid
		data.allNodes[newNodeId]->collapsedPixelIDs.push_back(frontier);
		data.allNodes[newNodeId]->parentsID = data.allNodes[frontier]->parentsID;
		data.allNodes[newNodeId]->childrenID = data.allNodes[frontier]->childrenID;
		vector<int> parentList;
		parentList.assign(data.allNodes[frontier]->parentsID.begin(), data.allNodes[frontier]->parentsID.end());
		for (int i = 0; i < parentList.size(); i++) {
			int parent = parentList[i];
			data.allNodes[parent]->childrenID.push_back(newNodeId);
			removeLink(data.allNodes[parent]->childrenID, frontier);
		}
		vector<int> childList;
		childList.assign(data.allNodes[frontier]->childrenID.begin(), data.allNodes[frontier]->childrenID.end());
		for (int i = 0; i < childList.size(); i++) {
			int child = childList[i];
			data.allNodes[child]->parentsID.push_back(newNodeId);
			removeLink(data.allNodes[child]->parentsID, frontier);
		}
		visited[frontier] = 1;
		visited[newNodeId] = 1;
		queue<int> que;
		que.push(frontier);

		while (!que.empty()) {
			int currentNode = que.front();
			que.pop();
			vector<int> parentList;
			parentList.assign(data.allNodes[currentNode]->parentsID.begin(), data.allNodes[currentNode]->parentsID.end());
			for (int i = 0; i < parentList.size(); i++) {
				int parent = parentList[i];
				if (!visited[parent]) {
					double parentHeight = data.allNodes[parent]->elevation;
					if (parentHeight == frontierHeight) {
						if (data.allNodes[newNodeId]->parentsID.size() + data.allNodes[parent]->parentsID.size() - 1 <= parameter.maxParentDegree) {
							//merge parent with new node
							data.allNodes[newNodeId]->collapsedPixelIDs.push_back(parent);
							//update new node parent children list 
							removeLink(data.allNodes[newNodeId]->parentsID, parent);
							data.allNodes[newNodeId]->parentsID.insert(data.allNodes[newNodeId]->parentsID.end(), data.allNodes[parent]->parentsID.begin(), data.allNodes[parent]->parentsID.end());
							data.allNodes[newNodeId]->childrenID.insert(data.allNodes[newNodeId]->childrenID.end(), data.allNodes[parent]->childrenID.begin(), data.allNodes[parent]->childrenID.end());
							removeLink(data.allNodes[newNodeId]->childrenID, newNodeId);

							//remove newNode from the child list of parent
							removeLink(data.allNodes[parent]->childrenID, newNodeId);

							//update parent's neighbour list 
							vector<int> pList;
							pList.assign(data.allNodes[parent]->parentsID.begin(), data.allNodes[parent]->parentsID.end());
							for (int pl = 0; pl < pList.size(); pl++) {
								int p = pList[pl];
								data.allNodes[p]->childrenID.push_back(newNodeId);
								removeLink(data.allNodes[p]->childrenID, parent);
							}
							vector<int> cList;
							cList.assign(data.allNodes[parent]->childrenID.begin(), data.allNodes[parent]->childrenID.end());
							for (int cl = 0; cl < cList.size(); cl++) {
								int c = cList[cl];
								data.allNodes[c]->parentsID.push_back(newNodeId);
								removeLink(data.allNodes[c]->parentsID, parent);
							}
							//push merged node into queue
							visited[parent] = 1;
							que.push(parent);
						}
					}
				}
			}

			vector<int> childList;
			childList.assign(data.allNodes[currentNode]->childrenID.begin(), data.allNodes[currentNode]->childrenID.end());
			for (int i = 0; i < childList.size(); i++) {
				int child = childList[i];
				if (!visited[child]) {
					double childHeight = data.allNodes[child]->elevation;
					if (childHeight == frontierHeight) {
						if (data.allNodes[newNodeId]->parentsID.size() + data.allNodes[child]->parentsID.size() - 1 <= parameter.maxParentDegree) {
							//merge child with new node
							data.allNodes[newNodeId]->collapsedPixelIDs.push_back(child);
							//update new node parent children list 
							removeLink(data.allNodes[newNodeId]->childrenID, child);
							data.allNodes[newNodeId]->childrenID.insert(data.allNodes[newNodeId]->childrenID.end(), data.allNodes[child]->childrenID.begin(), data.allNodes[child]->childrenID.end());
							data.allNodes[newNodeId]->parentsID.insert(data.allNodes[newNodeId]->parentsID.end(), data.allNodes[child]->parentsID.begin(), data.allNodes[child]->parentsID.end());
							removeLink(data.allNodes[newNodeId]->parentsID, newNodeId);

							//remove newNode from the parent list of child
							removeLink(data.allNodes[child]->parentsID, newNodeId);

							//update child's neighbour list 
							vector<int> pList;
							pList.assign(data.allNodes[child]->parentsID.begin(), data.allNodes[child]->parentsID.end());
							for (int pl = 0; pl < pList.size(); pl++) {
								int p = pList[pl];
								data.allNodes[p]->childrenID.push_back(newNodeId);
								removeLink(data.allNodes[p]->childrenID, child);
							}
							vector<int> cList;
							cList.assign(data.allNodes[child]->childrenID.begin(), data.allNodes[child]->childrenID.end());
							for (int cl = 0; cl < cList.size(); cl++) {
								int c = cList[cl];
								data.allNodes[c]->parentsID.push_back(newNodeId);
								removeLink(data.allNodes[c]->parentsID, child);
							}
							//push merged node into queue
							visited[child] = 1;
							que.push(child);
						}
					}
				}
			}
		}
		newNodeId++;
	}

	int collapseIndex = parameter.allPixelSize;
	int collapseEndIndex = newNodeId;
	int newCollapseNodeCount = collapseEndIndex - collapseIndex;
	for (int i = 0; i < newCollapseNodeCount; i++) {
		data.allNodes[i] = data.allNodes[collapseIndex];
		data.allNodes[i]->nodeIndex = i;
		for (int c = 0; c < data.allNodes[i]->childrenID.size(); c++) {
			data.allNodes[i]->childrenID[c] = data.allNodes[i]->childrenID[c] - parameter.allPixelSize;
		}
		for (int p = 0; p < data.allNodes[i]->parentsID.size(); p++) {
			data.allNodes[i]->parentsID[p] = data.allNodes[i]->parentsID[p] - parameter.allPixelSize;
		}
		//if (data.allNodes[i]->parentsID.size() > parameter.maxParentDegree /*|| data.allNodes[i]->childrenID.size()>3*/) {
		//	std::cout << "Node " << i << " Number of  Parents = " << data.allNodes[i]->parentsID.size() << " Number of Children = " << data.allNodes[i]->childrenID.size() << endl;
		//}
		collapseIndex++;
	}

	//reseting parameter for collapse tree
	parameter.allPixelSize = newCollapseNodeCount; //assumption: no missing values
												   //displayTree(3);
	data.allNodes.resize(parameter.allPixelSize);
	data.allNodes.shrink_to_fit();

	ofstream outfile;
	outfile.open(CTOutputLocation + "NodeDegree.txt");
	for (int i = 0; i < parameter.allPixelSize; i++) {
		outfile << data.allNodes[i]->collapsedPixelIDs.size() << "," << data.allNodes[i]->parentsID.size() << "," << data.allNodes[i]->childrenID.size() << endl;
	}
	outfile.close();
	//topologicalSort(); //find the new topological sort order for collapse tree does not work now because some of the child id is greater than allpixelsize
	//displayTree(3);
}
void cFlood::collapse() {
	int newNodeId = parameter.allPixelSize;
	vector<int> visited(parameter.allPixelSize * 2, 0);
	for (int tsIndex = 0; tsIndex < parameter.allPixelSize; tsIndex++) {
		int frontier = data.elevationIndexPair[tsIndex].second;/*topoSortedIndex[tsIndex];*/
		if (visited[frontier]) {
			continue;
		}
		double frontierHeight = data.allNodes[frontier]->elevation;
		data.allNodes.push_back(new Node(frontierHeight, newNodeId));  //maybe just push nodeid
		queue<int> que;
		que.push(frontier);
		while (!que.empty()) {
			int currentNode = que.front();
			visited[currentNode] += 1;
			que.pop();

			data.allNodes[newNodeId]->collapsedPixelIDs.push_back(currentNode);

			vector<int> parentList;
			parentList.assign(data.allNodes[currentNode]->parentsID.begin(), data.allNodes[currentNode]->parentsID.end());
			for (int i = 0; i < parentList.size(); i++) {
				int parent = parentList[i];
				if (!visited[parent]) {
					double parentHeight = data.allNodes[parent]->elevation;
					if (parentHeight != frontierHeight) {
						data.allNodes[parent]->childrenID.push_back(newNodeId);
						removeLink(data.allNodes[parent]->childrenID, currentNode);

						data.allNodes[newNodeId]->parentsID.push_back(parent);
						removeLink(data.allNodes[currentNode]->parentsID, parent);
					}
					else {
						//first check the number of uncollapsable parent in new node
						//confirm the number of uncollapsable parent size for new node is within  the limit
						if (data.allNodes[parent]->parentsID.size() == 1 && data.allNodes[parent]->collapsedPixelIDs.size() == 0) {
							que.push(parent);
							removeLink(data.allNodes[parent]->childrenID, currentNode);
							removeLink(data.allNodes[currentNode]->parentsID, parent);
						}
						else {
							data.allNodes[parent]->childrenID.push_back(newNodeId);
							removeLink(data.allNodes[parent]->childrenID, currentNode);

							data.allNodes[newNodeId]->parentsID.push_back(parent);
							removeLink(data.allNodes[currentNode]->parentsID, parent);
						}

					}
				}
			}
			//data.allNodes[currentNode]->parentsID.clear();

			vector<int> childList;
			childList.assign(data.allNodes[currentNode]->childrenID.begin(), data.allNodes[currentNode]->childrenID.end());
			for (int i = 0; i < childList.size(); i++) {
				int child = childList[i];
				if (!visited[child]) {
					double childHeight = data.allNodes[child]->elevation;
					if (childHeight != frontierHeight) {
						data.allNodes[child]->parentsID.push_back(newNodeId);
						data.allNodes[newNodeId]->childrenID.push_back(child);
						removeLink(data.allNodes[currentNode]->childrenID, child);
						removeLink(data.allNodes[child]->parentsID, currentNode);
					}
					else {
						//confirm the number of uncollapsable parent size for new node is within  the limit
						if (data.allNodes[child]->parentsID.size() == 1 && data.allNodes[child]->collapsedPixelIDs.size() == 0) {
							que.push(child);
							removeLink(data.allNodes[currentNode]->childrenID, child);
							removeLink(data.allNodes[child]->parentsID, currentNode);
						}
						else {
							data.allNodes[child]->parentsID.push_back(newNodeId);
							data.allNodes[newNodeId]->childrenID.push_back(child);
							removeLink(data.allNodes[currentNode]->childrenID, child);
							removeLink(data.allNodes[child]->parentsID, currentNode);
						}
					}
				}
			}
			//data.allNodes[currentNode]->childrenID.clear();
		}
		newNodeId++;
	}

	int collapseIndex = parameter.allPixelSize;
	int collapseEndIndex = newNodeId;
	int newCollapseNodeCount = collapseEndIndex - collapseIndex;
	for (int i = 0; i < newCollapseNodeCount; i++) {
		data.allNodes[i] = data.allNodes[collapseIndex];
		data.allNodes[i]->nodeIndex = i;
		for (int c = 0; c < data.allNodes[i]->childrenID.size(); c++) {
			data.allNodes[i]->childrenID[c] = data.allNodes[i]->childrenID[c] - parameter.allPixelSize;
		}
		for (int p = 0; p < data.allNodes[i]->parentsID.size(); p++) {
			data.allNodes[i]->parentsID[p] = data.allNodes[i]->parentsID[p] - parameter.allPixelSize;
		}
		if (data.allNodes[i]->parentsID.size() > 10 /*|| data.allNodes[i]->childrenID.size()>3*/) {
			cout << "Node " << i << " Number of  Parents = " << data.allNodes[i]->parentsID.size() << " Number of Children = " << data.allNodes[i]->childrenID.size() << endl;
		}
		collapseIndex++;
	}

	//reseting parameter for collapse tree
	parameter.allPixelSize = newCollapseNodeCount; //assumption: no missing values
												   //displayTree(3);
	data.allNodes.resize(parameter.allPixelSize);
	data.allNodes.shrink_to_fit();
	//topologicalSort(); //find the new topological sort order for collapse tree does not work now because some of the child id is greater than allpixelsize
	int CollapsedPixelsCounter = 0;
	for (int i = 0; i < parameter.allPixelSize; i++) {
		CollapsedPixelsCounter += (int)data.allNodes[i]->collapsedPixelIDs.size();
		if (data.allNodes[i]->childrenID.size() == 0 && data.allNodes[i]->parentsID.size() == 0) {
			cout << "Hanging Nodes " << i << endl;
		}
		if (data.allNodes[i]->parentsID.size() > 10) {
			cout << "Max Parents Nodes " << i << "Parent Count" << data.allNodes[i]->parentsID.size() << endl;
		}
	}
	cout << "parameter.orgPixelSize = " << parameter.orgPixelSize << endl;
	cout << "CollapsedPixelsCounter = " << CollapsedPixelsCounter << endl;

}

void cFlood::inference() {
	vector<int> inferVisited(parameter.allPixelSize, 0);
	for (int i = 0; i < parameter.allPixelSize; i++) {
		data.allNodes[i]->correspondingNeighbour.clear();
		data.allNodes[i]->correspondingNeighbourClassOne.clear();
		data.allNodes[i]->correspondingNeighbourClassZero.clear();
	}
	//data.allNodes.correspondingNeighboursClass.resize(parameter.allPixelSize * cNum);
	//infer.correspondingNeighbours.resize(parameter.allPixelSize);
	int bfsTraversalOrderSize = (int)data.bfsTraversalOrder.size();
	for (int node = bfsTraversalOrderSize - 1; node >= 0; node--) {
		int cur_node_id = data.bfsTraversalOrder[node];
		data.allNodes[cur_node_id]->fi_ChildList.resize(data.allNodes[cur_node_id]->childrenID.size()*cNum, 0);
		for (int cls = 0; cls < cNum; cls++) {
			data.allNodes[cur_node_id]->fi_parent[cls] = 0;
			data.allNodes[cur_node_id]->fo[cls] = 0;
		}

		//first figure out which neighbor fmessage passes to from current node pass n->? foNode;
		//idea: In bfs traversal order leave to root, check if next the node in bfs order is parent or child of the current node (should be child or parent of the current node)
		int foNode = -1;
		bool foNode_isChild = false;
		for (int p = 0; p < data.allNodes[cur_node_id]->parentsID.size(); p++) {  //check in parent list if found respective parent node is foNode
			int pid = data.allNodes[cur_node_id]->parentsID[p];
			if (!inferVisited[pid]) {
				foNode = pid;
				break;
			}
		}
		if (foNode == -1) {
			for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
				int cid = data.allNodes[cur_node_id]->childrenID[c];
				if (!inferVisited[cid]) {
					foNode = cid;
					foNode_isChild = true;
					break;
				}
			}
		}
		data.allNodes[cur_node_id]->foNode = foNode;
		data.allNodes[cur_node_id]->foNode_ischild = foNode_isChild;
		if (cur_node_id == parameter.bfsRoot && data.allNodes[cur_node_id]->childrenID.size() == 0) {
			foNode_isChild = true;
		}

		//incoming message from visited child
		if (data.allNodes[cur_node_id]->childrenID.size() > 0) {

			for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
				int child_id = data.allNodes[cur_node_id]->childrenID[c];

				if (child_id == foNode) {
					continue;
				}
				data.allNodes[cur_node_id]->correspondingNeighbour.push_back(child_id);
				for (int p = 0; p < data.allNodes[child_id]->parentsID.size(); p++) {
					int pid = data.allNodes[child_id]->parentsID[p];
					if (pid != cur_node_id) {
						data.allNodes[cur_node_id]->correspondingNeighbour.push_back(pid);
					}

				}
				vector<int> parentOfChildExcept_currentNode;
				for (int en = 0; en < data.allNodes[child_id]->parentsID.size(); en++) {
					if (data.allNodes[child_id]->parentsID[en] != cur_node_id) {
						parentOfChildExcept_currentNode.push_back(data.allNodes[child_id]->parentsID[en]);
					}

				}
				for (int cls = 0; cls < cNum; cls++) {  //cls represents current node class
														//double sumAccumulator = eln(0);   //should be 0 since we are summing it up//eln(1);//need to confirm
					double max = eln(0);
					vector<int> maxCorrespondingNeighbour;
					for (int c_cls = 0; c_cls < cNum; c_cls++) { //c_cls reperesnets child class label   Yc
						int max_bitCount = 1 << parentOfChildExcept_currentNode.size();
						for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent and child class label(given by c_cls)
							double productAccumulator = data.allNodes[child_id]->fo[c_cls];  //product with fo(c)
							vector<int>neighbourClass;
							neighbourClass.push_back(c_cls);
							int parentClsProd = 1; //p(c), product of parent classes for child c
							for (int p = 0; p < parentOfChildExcept_currentNode.size(); p++) {//calculating Product(fo(p)) for all parent of current child except the current node
								int pid = parentOfChildExcept_currentNode[p];
								int parentClsValue = (bitCount >> p) & 1;
								parentClsProd *= parentClsValue;
								neighbourClass.push_back(parentClsValue);
								productAccumulator = elnproduct(productAccumulator, data.allNodes[pid]->fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
							}
							//multiplying P(Yc|Ypc)
							parentClsProd *= cls;
							productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[c_cls][parentClsProd]);
							if (max < productAccumulator) {
								max = productAccumulator;
								maxCorrespondingNeighbour = neighbourClass;
							}
						}
					}
					data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls] = max;
					if (cls == 0) {
						for (int t = 0; t < maxCorrespondingNeighbour.size(); t++) {
							data.allNodes[cur_node_id]->correspondingNeighbourClassZero.push_back(maxCorrespondingNeighbour[t]);
						}
					}
					else {
						for (int t = 0; t < maxCorrespondingNeighbour.size(); t++) {
							data.allNodes[cur_node_id]->correspondingNeighbourClassOne.push_back(maxCorrespondingNeighbour[t]);
						}
					}
				}
			}
		}

		if (foNode_isChild) {  //means the current node has all visited parents
			if (data.allNodes[cur_node_id]->parentsID.size() == 0) {
				for (int cls = 0; cls < cNum; cls++) {
					data.allNodes[cur_node_id]->fi_parent[cls] = parameter.elnPz[cls];
				}
			}
			else {
				for (int p = 0; p < data.allNodes[cur_node_id]->parentsID.size(); p++) {
					int pid = data.allNodes[cur_node_id]->parentsID[p];
					data.allNodes[cur_node_id]->correspondingNeighbour.push_back(pid);
				}
				for (int cls = 0; cls < cNum; cls++) {
					double max = eln(0);
					vector<int> maxNeighbourClass;
					int max_bitCount = 1 << data.allNodes[cur_node_id]->parentsID.size();
					for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent class label
						vector<int> parentClass;
						double productAccumulator = eln(1);
						int parentClsProd = 1;
						for (int p = 0; p < data.allNodes[cur_node_id]->parentsID.size(); p++) {
							int pid = data.allNodes[cur_node_id]->parentsID[p];
							int parentClsValue = (bitCount >> p) & 1;
							parentClass.push_back(parentClsValue);
							parentClsProd *= parentClsValue;
							productAccumulator = elnproduct(productAccumulator, data.allNodes[pid]->fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
						}
						productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[cls][parentClsProd]);
						if (max < productAccumulator) {
							max = productAccumulator;
							maxNeighbourClass = parentClass;
						}
						//sumAccumulator = elnsum(sumAccumulator, productAccumulator);
					}
					data.allNodes[cur_node_id]->fi_parent[cls] = max;
					if (cls == 0) {
						for (int t = 0; t < maxNeighbourClass.size(); t++) {
							data.allNodes[cur_node_id]->correspondingNeighbourClassZero.push_back(maxNeighbourClass[t]);
						}
					}
					else {
						for (int t = 0; t < maxNeighbourClass.size(); t++) {
							data.allNodes[cur_node_id]->correspondingNeighbourClassOne.push_back(maxNeighbourClass[t]);
						}
					}
				}
			}

			//calulating fo
			for (int cls = 0; cls < cNum; cls++) { //cls represents class of the current node
				double productAccumulator = eln(1);
				for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
					int child_id = data.allNodes[cur_node_id]->childrenID[c];
					if (child_id == foNode) {
						continue;
					}
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls]); //multiplying with al the child fi except the outgoing child
				}
				productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_parent[cls]);  // multiplying with fi(n)_parent
				productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[cur_node_id*cNum + cls]);

				data.allNodes[cur_node_id]->fo[cls] = productAccumulator;
			}

		}

		else {  //message pass n-> parent there is no fi(n)_parent   //computes for root node as well
				//calulating fo
			for (int cls = 0; cls < cNum; cls++) { //cls represents class of the current node
				double productAccumulator = eln(1);
				for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls]); //multiplying with al the child fi except the outgoing child
				}
				productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[cur_node_id*cNum + cls]);
				data.allNodes[cur_node_id]->fo[cls] = productAccumulator;
			}
		}

		inferVisited[cur_node_id] = 1;
	}
	updateMapPrediction();
}
void cFlood::updateMapPrediction() {
	//extracting class
	vector<int> nodeClass(parameter.allPixelSize, -1);
	mappredictions.resize(parameter.orgPixelSize, -1);
	queue<int> que;
	int nodeCls;
	if (data.allNodes[parameter.bfsRoot]->fo[0] > data.allNodes[parameter.bfsRoot]->fo[1]) {
		nodeCls = 0;
	}
	else {
		nodeCls = 1;
	}
	nodeClass[parameter.bfsRoot] = nodeCls;
	que.push(parameter.bfsRoot);
	while (!que.empty()) {
		int node = que.front();
		que.pop();
		int nodeCls = nodeClass[node];
		for (int c = 0; c < data.allNodes[node]->correspondingNeighbour.size(); c++) {
			int neigh_id = data.allNodes[node]->correspondingNeighbour[c];
			int cClass;
			if (nodeCls == 0) {
				cClass = data.allNodes[node]->correspondingNeighbourClassZero[c];
			}
			else {
				cClass = data.allNodes[node]->correspondingNeighbourClassOne[c];
			}
			nodeClass[neigh_id] = cClass;
			que.push(neigh_id);
		}
	}


	if (parameter.CollapseSwitch) {
		for (int i = 0; i < nodeClass.size(); i++) {
			for (int cp = 0; cp < data.allNodes[i]->collapsedPixelIDs.size(); cp++) {
				int id = data.allNodes[i]->collapsedPixelIDs[cp];
				mappredictions[id] = nodeClass[i];
			}
		}
	}
	else {
		mappredictions = nodeClass;
	}
}
void cFlood::output() {
	auto start = std::chrono::system_clock::now();
	ofstream classout;
	classout.open(CTOutputLocation + CTPrediction);
	for (int i = 0; i < mappredictions.size(); i++) {
		classout << mappredictions[i] << endl;
	}
	classout.close();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double>elapsed_seconds = end - start;
	cout << "Writing Prediction File took " << elapsed_seconds.count() << "seconds" << endl;
}

void cFlood::Collapselearning() {

	infer.marginal_ZnZpn.resize(parameter.allPixelSize * cNum * cNum, LOGZERO); // All except bottom nodes
	infer.marginal_Zn.resize(parameter.allPixelSize * cNum, LOGZERO); // Marginal Zn

	int iterateTimes = 0;
	bool iterator = true;

	double PiOld, EpsilonOld;
	double MuOld[cNum][Dim], SigmaOld[cNum][Dim][Dim];
	while (iterator) {
		this->UpdateTransProb();

		//copy current parameters to compare across iterations
		PiOld = parameter.Pi;
		EpsilonOld = parameter.Epsilon;
		for (int c = 0; c < cNum; c++) {
			for (int i = 0; i < Dim; i++) {
				MuOld[c][i] = parameter.Mu[c][i];
				for (size_t j = 0; j < Dim; j++) {
					SigmaOld[c][i][j] = parameter.Sigma[c][i][j];
				}
			}
		}
		this->MessagePropagation();

		this->UpdateMarginalProb();

		this->CollapseUpdateParameters();

		this->CollapseUpdatePX_Z();

		//check stop criteria
		{
			bool MuConverge = true, SigmaConverge = true;
			double thresh = parameter.THRESHOLD;

			for (int c = 0; c < cNum; c++) {
				for (int i = 0; i < Dim; i++) {
					if (fabs((parameter.Mu[c][i] - MuOld[c][i]) / MuOld[c][i]) > thresh) {
						MuConverge = false;
						break;
					}

					for (int j = 0; j < Dim; j++) {
						if (fabs((parameter.Sigma[c][i][j] - SigmaOld[c][i][j]) / SigmaOld[c][i][j]) > thresh) {
							SigmaConverge = false;
							break;
						}
					}
				}
			}

			double epsilonRatio = fabs((eexp(parameter.Epsilon) - eexp(EpsilonOld)) / eexp(EpsilonOld));
			double PiRatio = fabs((eexp(parameter.Pi) - eexp(PiOld)) / eexp(PiOld));
			//double MRatio = fabs((eexp(parameter.M) - eexp(MOld)) / eexp(MOld));

			if (epsilonRatio < thresh &&  PiRatio < thresh && MuConverge && SigmaConverge) {
				iterator = false;
			}

			iterateTimes++;
			cout << "Iteration " << iterateTimes << " Done.." << endl;
			if (iterateTimes >= parameter.maxIteratTimes) {
				iterator = false;
			}
		}

	} // end while

}


void cFlood::learning() {
	infer.marginal_ZnZpn.resize(parameter.allPixelSize * cNum * cNum); // All except bottom nodes
	infer.marginal_Zn.resize(parameter.allPixelSize * cNum); // Marginal Zn

	int iterateTimes = 0;
	bool iterator = true;

	double PiOld, EpsilonOld;
	double MuOld[cNum][Dim], SigmaOld[cNum][Dim][Dim];

	while (iterator) {
		this->UpdateTransProb();

		//copy current parameters to compare across iterations
		PiOld = parameter.Pi;
		EpsilonOld = parameter.Epsilon;
		for (int c = 0; c < cNum; c++) {
			for (int i = 0; i < Dim; i++) {
				MuOld[c][i] = parameter.Mu[c][i];
				for (size_t j = 0; j < Dim; j++) {
					SigmaOld[c][i][j] = parameter.Sigma[c][i][j];
				}
			}
		}
		this->MessagePropagation();

		this->UpdateMarginalProb();

		this->UpdateParameters();

		this->UpdatePX_Z();

		//check stop criteria
		{
			bool MuConverge = true, SigmaConverge = true;
			double thresh = parameter.THRESHOLD;

			for (int c = 0; c < cNum; c++) {
				for (int i = 0; i < Dim; i++) {
					if (fabs((parameter.Mu[c][i] - MuOld[c][i]) / MuOld[c][i]) > thresh) {
						MuConverge = false;
						break;
					}

					for (int j = 0; j < Dim; j++) {
						if (fabs((parameter.Sigma[c][i][j] - SigmaOld[c][i][j]) / SigmaOld[c][i][j]) > thresh) {
							SigmaConverge = false;
							break;
						}
					}
				}
			}

			double epsilonRatio = fabs((eexp(parameter.Epsilon) - eexp(EpsilonOld)) / eexp(EpsilonOld));
			double PiRatio = fabs((eexp(parameter.Pi) - eexp(PiOld)) / eexp(PiOld));
			//double MRatio = fabs((eexp(parameter.M) - eexp(MOld)) / eexp(MOld));

			if (epsilonRatio < thresh &&  PiRatio < thresh && MuConverge && SigmaConverge) {
				iterator = false;
			}

			iterateTimes++;
			if (iterateTimes >= parameter.maxIteratTimes) {
				iterator = false;
			}
		}

	} // end while

}

void cFlood::UpdateParameters() {

	//// Calculate new parameter
	double topEpsilon = LOGZERO, bottomEpsilon = LOGZERO, topPi = LOGZERO, bottomPi = LOGZERO;
	double bottomMu[cNum] = { LOGZERO };
	double tempMu[cNum][Dim] = { LOGZERO };
	double xMinusMu[cNum][Dim];
	double SigmaTemp[cNum][Dim][Dim] = { 0 };

	for (int i = 0; i < parameter.allPixelSize; i++) {
		int curIdx = i;

		//// M,  go through all nodes
		//for (int z = 0; z < cNum; z++) {
		//	for (int y = 0; y < cNum; y++) {
		//		topM = elnsum(topM, elnproduct(eln(z * (1 - y)), infer.marginal_YnZn[i*cNum*cNum + y*cNum + z]));
		//		bottomM = elnsum(bottomM, elnproduct(eln(z), infer.marginal_YnZn[i*cNum*cNum + y*cNum + z]));
		//	}
		//}
		////topM = elnsum(topM, infer.marginal_YnZn[i * cNum]);
		////bottomM = elnsum(bottomM, elnsum(infer.marginal_YnZn[i * cNum], infer.marginal_YnZn[i * cNum + 1]));

		// Epsilon, zi has parents
		if (data.allNodes[curIdx]->parentsID.size() > 0) {
			for (int z = 0; z < cNum; z++) {
				for (int zp = 0; zp < cNum; zp++) {
					topEpsilon = elnsum(topEpsilon, elnproduct(eln(zp*(1 - z)), infer.marginal_ZnZpn[i * cNum*cNum + z * cNum + zp]));
					bottomEpsilon = elnsum(bottomEpsilon, elnproduct(eln(zp), infer.marginal_ZnZpn[i * cNum*cNum + z * cNum + zp]));
				}
			}
			//topEpsilon = elnsum(topEpsilon, infer.marginal_ZnZpn[i * cNum]);
			//bottomEpsilon = elnsum(bottomEpsilon, elnsum(infer.marginal_ZnZpn[i * cNum], infer.marginal_ZnZpn[i * cNum + 1]));
		}
		// Pi, zi is leaf node
		else {
			for (int z = 0; z < cNum; z++) {
				topPi = elnsum(topPi, elnproduct(eln(1 - z), infer.marginal_Zn[i * cNum + z]));
				bottomPi = elnsum(bottomPi, infer.marginal_Zn[i * cNum + z]);
			}

			//for (int z = 0; z < cNum; z++) {
			//	topPi = elnsum(topPi, elnproduct( eln(1-z), infer.marginal_ZnZpn[i * cNum*cNum + z*cNum]));
			//	bottomPi = elnsum(bottomPi, infer.marginal_ZnZpn[i * cNum*cNum + z*cNum]);
			//}
			//topPi = elnsum(topPi, infer.marginal_ZnZpn[i * cNum]);
			//bottomPi = elnsum(bottomPi, elnsum(infer.marginal_ZnZpn[i * cNum], infer.marginal_ZnZpn[i * cNum + 1]));
		}

		// Mu0, Mu1, go through all nodes
		for (size_t j = 0; j < Dim; j++) {
			for (int c = 0; c < cNum; c++) {
				tempMu[c][j] = elnsum(tempMu[c][j], elnproduct(eln(data.features[i * Dim + j]), infer.marginal_Zn[i * cNum + c]));
			}
		}
		for (int c = 0; c < cNum; c++) {
			bottomMu[c] = elnsum(bottomMu[c], infer.marginal_Zn[i * cNum + c]);
		}
	}

	parameter.Epsilon = elnproduct(topEpsilon, -1 * bottomEpsilon);
	parameter.Pi = elnproduct(topPi, -1 * bottomPi);


	// reserve eln(Mu) form
	for (size_t j = 0; j < Dim; j++) {
		for (int c = 0; c < cNum; c++) {
			parameter.elnMu[c][j] = elnproduct(tempMu[c][j], -1 * bottomMu[c]);
		}
	}

	// convert Mu to normal
	for (size_t j = 0; j < Dim; j++) {
		for (int c = 0; c < cNum; c++) {
			parameter.Mu[c][j] = eexp(parameter.elnMu[c][j]);
		}
	}


	// Update Sigma
	for (size_t i = 0; i < parameter.allPixelSize; i++) {

		for (int c = 0; c < cNum; c++) {
			for (size_t j = 0; j < Dim; j++) {
				xMinusMu[c][j] = data.features[i * Dim + j] - parameter.Mu[c][j];
			}
		}

		for (int c = 0; c < cNum; c++) {
			for (size_t m = 0; m < Dim; m++) { // row
				for (size_t n = 0; n < Dim; n++) { // column
					SigmaTemp[c][m][n] += xMinusMu[c][m] * xMinusMu[c][n] * eexp(infer.marginal_Zn[i * cNum + c]);
				}
			}
		}
	}

	for (int c = 0; c < cNum; c++) {
		for (size_t i = 0; i < Dim; i++) {
			for (size_t j = 0; j < Dim; j++) {
				parameter.Sigma[c][i][j] = SigmaTemp[c][i][j] / eexp(bottomMu[c]); // bottom is the same as Mu
			}
		}
	}

}


void cFlood::CollapseUpdateParameters() {

	//// Calculate new parameter
	double topEpsilon = LOGZERO, bottomEpsilon = LOGZERO, topPi = LOGZERO, bottomPi = LOGZERO;
	double bottomMu[cNum] = { LOGZERO };
	double tempMu[cNum][Dim] = { LOGZERO };
	double SigmaTemp[cNum][Dim][Dim] = { 0 };
	double MuMuT[cNum][Dim][Dim] = { 0 };

	for (int i = 0; i < parameter.allPixelSize; i++) {
		int curIdx = i;
		// Epsilon, zi has parents
		if (data.allNodes[curIdx]->parentsID.size() > 0) {
			for (int z = 0; z < cNum; z++) {
				for (int zp = 0; zp < cNum; zp++) {
					topEpsilon = elnsum(topEpsilon, elnproduct(eln(zp*(1 - z)), infer.marginal_ZnZpn[i * cNum*cNum + z * cNum + zp]));
					bottomEpsilon = elnsum(bottomEpsilon, elnproduct(eln(zp), infer.marginal_ZnZpn[i * cNum*cNum + z * cNum + zp]));
				}
			}
			//topEpsilon = elnsum(topEpsilon, infer.marginal_ZnZpn[i * cNum]);
			//bottomEpsilon = elnsum(bottomEpsilon, elnsum(infer.marginal_ZnZpn[i * cNum], infer.marginal_ZnZpn[i * cNum + 1]));
		}
		// Pi, zi is leaf node
		else {
			for (int z = 0; z < cNum; z++) {
				topPi = elnsum(topPi, elnproduct(eln(1 - z), infer.marginal_Zn[i * cNum + z]));
				bottomPi = elnsum(bottomPi, infer.marginal_Zn[i * cNum + z]);
			}

			//for (int z = 0; z < cNum; z++) {
			//	topPi = elnsum(topPi, elnproduct( eln(1-z), infer.marginal_ZnZpn[i * cNum*cNum + z*cNum]));
			//	bottomPi = elnsum(bottomPi, infer.marginal_ZnZpn[i * cNum*cNum + z*cNum]);
			//}
			//topPi = elnsum(topPi, infer.marginal_ZnZpn[i * cNum]);
			//bottomPi = elnsum(bottomPi, elnsum(infer.marginal_ZnZpn[i * cNum], infer.marginal_ZnZpn[i * cNum + 1]));
		}

		// Mu0, Mu1, go through all nodes
		for (size_t j = 0; j < Dim; j++) {
			for (int c = 0; c < cNum; c++) {
				tempMu[c][j] = elnsum(tempMu[c][j], elnproduct(eln(data.B[i * Dim + j]), infer.marginal_Zn[i * cNum + c]));
			}
		}
		for (int c = 0; c < cNum; c++) {
			bottomMu[c] = elnsum(bottomMu[c], elnproduct(infer.marginal_Zn[i * cNum + c], eln(data.C[i])));
		}
	}

	parameter.Epsilon = elnproduct(topEpsilon, -1 * bottomEpsilon);
	//parameter.Epsilon = eln(0.00001);
	parameter.Pi = elnproduct(topPi, -1 * bottomPi);


	// reserve eln(Mu) form
	for (size_t j = 0; j < Dim; j++) {
		for (int c = 0; c < cNum; c++) {
			parameter.elnMu[c][j] = elnproduct(tempMu[c][j], -1 * bottomMu[c]);
		}
	}

	// convert Mu to normal
	for (size_t j = 0; j < Dim; j++) {
		for (int c = 0; c < cNum; c++) {
			parameter.Mu[c][j] = eexp(parameter.elnMu[c][j]);
		}
	}

	//now parameter.Mu are normal, not extended log values
	for (int c = 0; c < cNum; c++) {
		for (size_t m = 0; m < Dim; m++) { // row
			for (size_t n = 0; n < Dim; n++) { // column
				MuMuT[c][m][n] = parameter.Mu[c][m] * parameter.Mu[c][n];
			}
		}
	}

	//// Update Sigma
	//for (size_t i = 0; i < parameter.allPixelSize; i++) {
	//	for (int c = 0; c < cNum; c++) {
	//		for (size_t m = 0; m < Dim; m++) { // row
	//			for (size_t n = 0; n < Dim; n++) { // column
	//				double tmp_sum = data.A[i * Dim * Dim + m * Dim + n] - data.B[i * Dim + m] * parameter.Mu[c][n]
	//					- data.B[i * Dim + n] * parameter.Mu[c][m] + data.C[i] * MuMuT[c][m][n];
	//				//SigmaTemp[c][m][n] += tmp_sum * eexp(infer.marginal_Zn[i * cNum + c]);
	//				SigmaTemp[c][m][n] = elnsum(SigmaTemp[c][m][n], elnproduct(infer.marginal_Zn[i * cNum + c], eln(tmp_sum)));
	//			}
	//		}
	//	}
	//}


	//for (int c = 0; c < cNum; c++) {
	//	for (size_t i = 0; i < Dim; i++) {
	//		for (size_t j = 0; j < Dim; j++) {
	//			//parameter.Sigma[c][i][j] = SigmaTemp[c][i][j] / eexp(bottomMu[c]); // bottom is the same as Mu
	//			parameter.Sigma[c][i][j] = elnproduct(SigmaTemp[c][i][j], -1 * bottomMu[c]);
	//			parameter.Sigma[c][i][j] = eexp(parameter.Sigma[c][i][j]);
	//		}
	//	}
	//}

	// Update Sigma
	for (size_t i = 0; i < parameter.allPixelSize; i++) {

		for (int c = 0; c < cNum; c++) {
			for (size_t m = 0; m < Dim; m++) { // row
				for (size_t n = 0; n < Dim; n++) { // column
					double tmp_sum = data.A[i * Dim * Dim + m * Dim + n] - data.B[i * Dim + m] * parameter.Mu[c][n]
						- data.B[i * Dim + n] * parameter.Mu[c][m] + data.C[i] * MuMuT[c][m][n];
					SigmaTemp[c][m][n] += tmp_sum * eexp(infer.marginal_Zn[i * cNum + c]);
				}
			}
		}
	}

	for (int c = 0; c < cNum; c++) {
		for (size_t i = 0; i < Dim; i++) {
			for (size_t j = 0; j < Dim; j++) {
				parameter.Sigma[c][i][j] = SigmaTemp[c][i][j] / eexp(bottomMu[c]); // bottom is the same as Mu
			}
		}
	}

}

//the code assumes 2 by 2 transition matrix P(zn|zpn)
void cFlood::UpdateMarginalProb() {
	// Calculate Marginal distribution

	int curIdx;
	Node* curNode;
	vector<int>ParentOfZ(4);
	vector<int>parentIndexes(4);
	//vector<double>tempTerms(8), tempTermsNew(8);
	//double message0 = 0, message1 = 0;
	double normFactor;

	for (int i = 0; i < parameter.allPixelSize; i++) {
		curIdx = i;
		curNode = data.allNodes[curIdx];
		//initialize the result variable for cumulation
		for (int zn = 0; zn < cNum; zn++) {  //must initialize 
			for (int zpn = 0; zpn < cNum; zpn++) {
				infer.marginal_ZnZpn[curIdx*cNum*cNum + zn * cNum + zpn] = LOGZERO;
			}
		}
		// p(z, zp|X, theta) = multiply all outgoing messages towards the factor node (zn|zpn) * p(z|zp)
		// don't forget marginalization over Zpn
		if (curNode->parentsID.size() > 0) {

			if (data.allNodes[curIdx]->go_fromParent) {
				for (int cls = 0; cls < cNum; cls++) {
					int max_bitCount = 1 << data.allNodes[curIdx]->parentsID.size();
					for (int bitCount = 0; bitCount < max_bitCount; bitCount++) {
						double curMessage = data.allNodes[curIdx]->fo[cls];
						int parentClsProd = 1; //p(c), product of parent classes for child c

						for (int p = 0; p < data.allNodes[curIdx]->parentsID.size(); p++) {
							int pid = data.allNodes[curIdx]->parentsID[p];
							int parentClsValue = (bitCount >> p) & 1;
							parentClsProd *= parentClsValue;
							//maintain curMessage with go/fo on parent p
							if (pid == data.allNodes[curIdx]->go_lastVisitedNode) { //Po
								for (int c = 0; c < data.allNodes[pid]->childrenID.size(); c++) {
									if (data.allNodes[pid]->childrenID[c] == curIdx) {
										curMessage = elnproduct(curMessage, data.allNodes[pid]->go_ChildList[c*cNum + parentClsValue]);
										break;
									}
								}
							}
							else {
								curMessage = elnproduct(curMessage, data.allNodes[pid]->fo[parentClsValue]);
							}
						}
						curMessage = elnproduct(curMessage, parameter.elnPz_zpn[cls][parentClsProd]);
						infer.marginal_ZnZpn[curIdx*cNum*cNum + cls * cNum + parentClsProd] = elnsum(infer.marginal_ZnZpn[curIdx*cNum*cNum + cls * cNum + parentClsProd], curMessage);
					}
				}
			}
			else {  //when go is from child or child of parent
				for (int cls = 0; cls < cNum; cls++) {
					//double curMessage = eln(1); //eln(1);
					int max_bitCount = 1 << data.allNodes[curIdx]->parentsID.size();
					for (int bitCount = 0; bitCount < max_bitCount; bitCount++) {
						double curMessage = data.allNodes[curIdx]->go_parent[cls];
						int parentClsProd = 1; //p(c), product of parent classes for child c

						for (int p = 0; p < data.allNodes[curIdx]->parentsID.size(); p++) {
							int pid = data.allNodes[curIdx]->parentsID[p];
							int parentClsValue = (bitCount >> p) & 1;
							parentClsProd *= parentClsValue;
							//maintain curInMessage with go/fo on parent p
							curMessage = elnproduct(curMessage, data.allNodes[pid]->fo[parentClsValue]);
						}
						curMessage = elnproduct(curMessage, parameter.elnPz_zpn[cls][parentClsProd]);
						infer.marginal_ZnZpn[curIdx*cNum*cNum + cls * cNum + parentClsProd] = elnsum(infer.marginal_ZnZpn[curIdx*cNum*cNum + cls * cNum + parentClsProd], curMessage);
					}
				}

			}
			normFactor = LOGZERO;
			for (int zn = 0; zn < cNum; zn++) {
				for (int zpn = 0; zpn < cNum; zpn++) {
					normFactor = elnsum(normFactor, infer.marginal_ZnZpn[curIdx*cNum*cNum + zn * cNum + zpn]);
				}
			}

			//marginal_ZnZpn select the first and last term for each z
			for (int zn = 0; zn < cNum; zn++) {
				for (int zpn = 0; zpn < cNum; zpn++) {
					infer.marginal_ZnZpn[curIdx*cNum*cNum + zn * cNum + zpn] = elnproduct(infer.marginal_ZnZpn[curIdx*cNum*cNum + zn * cNum + zpn], -1 * normFactor);
				}
			}

			//verifying Marginal Probabiltiy sum should be equal to 1 
			//value should be in range (-inf,0]
			double sumTest = 0;
			for (int zn = 0; zn < cNum; zn++) {
				for (int zpn = 0; zpn < cNum; zpn++) {
					sumTest += eexp(infer.marginal_ZnZpn[curIdx*cNum*cNum + zn * cNum + zpn]);
					if (infer.marginal_ZnZpn[curIdx*cNum*cNum + zn * cNum + zpn] > 0) {
						cout << "Error in Marginal Probability Computation Node " << curIdx << " Zn= " << zn << " Zpn= " << zpn << endl;
					}
				}
			}
			if (abs(sumTest - 1) > 0.0001) {
				cout << "Error in Marginal Probability Computation Node " << curIdx << " Sum is not equal to 1" << endl;
			}

		}

		//else {
		// P(z|X, theta) = gi * fi * vi, Marginal Zn
		normFactor = LOGZERO;
		if (data.allNodes[curIdx]->go_fromParent) {
			for (int z = 0; z < cNum; z++) {
				//compute infer.marginal_Zn[curIdx * cNum + z] based on all incoming messages to the node, including gi's, fi's, and P(xn|Zn)
				//infer.marginal_Zn[curIdx * cNum + z] = elnproduct(elnproduct(infer.lnfi[curIdx * cNum + z], infer.lngi[curIdx * cNum + z]), infer.lnvi[curIdx * cNum + z]);
				double curMessage = data.allNodes[curIdx]->gi[z];
				//incoming from the child side
				for (int c = 0; c < data.allNodes[curIdx]->childrenID.size(); c++) {
					curMessage = elnproduct(curMessage, data.allNodes[curIdx]->fi_ChildList[c*cNum + z]);
				}
				curMessage = elnproduct(curMessage, parameter.elnPxn_zn[curIdx*cNum + z]);
				infer.marginal_Zn[curIdx * cNum + z] = curMessage;
				normFactor = elnsum(normFactor, infer.marginal_Zn[curIdx * cNum + z]);
			}
		}
		else { //when go is from child or child of parent
			for (int z = 0; z < cNum; z++) {
				//compute infer.marginal_Zn[curIdx * cNum + z] based on all incoming messages to the node, including gi's, fi's, and P(xn|Zn)
				//infer.marginal_Zn[curIdx * cNum + z] = elnproduct(elnproduct(infer.lnfi[curIdx * cNum + z], infer.lngi[curIdx * cNum + z]), infer.lnvi[curIdx * cNum + z]);
				double curMessage = data.allNodes[curIdx]->fi_parent[z];
				curMessage = elnproduct(curMessage, data.allNodes[curIdx]->gi[z]);
				//incoming from the child side
				for (int c = 0; c < data.allNodes[curIdx]->childrenID.size(); c++) {
					if (data.allNodes[curIdx]->childrenID[c] == data.allNodes[curIdx]->go_lastVisitedNode) {
						continue;
					}
					curMessage = elnproduct(curMessage, data.allNodes[curIdx]->fi_ChildList[c*cNum + z]);
				}
				curMessage = elnproduct(curMessage, parameter.elnPxn_zn[curIdx*cNum + z]);
				infer.marginal_Zn[curIdx * cNum + z] = curMessage;
				normFactor = elnsum(normFactor, infer.marginal_Zn[curIdx * cNum + z]);
			}
		}
		//}
		for (int z = 0; z < cNum; z++) {
			infer.marginal_Zn[curIdx * cNum + z] = elnproduct(infer.marginal_Zn[curIdx * cNum + z], -1 * normFactor);
		}

		double sumTest = 0;
		for (int c = 0; c < cNum; c++) {
			sumTest += eexp(infer.marginal_Zn[curIdx * cNum + c]);
			if (infer.marginal_Zn[curIdx * cNum + c] > 0) {
				cout << "wrong message: marginal_Zn" << endl;
			}
		}
		if (abs(sumTest - 1) > 0.0001) {
			cout << "Error in Marginal Probability Computation Node " << curIdx << " Sum is not equal to 1" << endl;
		}
	}
}

//Assume the first node is node without parents 
void cFlood::MessagePropagation() {
	//NOTE: we can only handle 64 parents/children for long int bit_counter;
	//sort all nodes in BFS traversal order	 
	vector<int> mpVisited(parameter.allPixelSize, 0);
	//leaves to root
	int bfsTraversalOrderSize = (int)data.bfsTraversalOrder.size();
	for (int node = bfsTraversalOrderSize - 1; node >= 0; node--) {
		int cur_node_id = data.bfsTraversalOrder[node];  //n
		//initializing fi_childlist,fi_parent fo
		data.allNodes[cur_node_id]->fi_ChildList.resize(data.allNodes[cur_node_id]->childrenID.size()*cNum, 0);
		for (int cls = 0; cls < cNum; cls++) {
			data.allNodes[cur_node_id]->fi_parent[cls] = 0;
			data.allNodes[cur_node_id]->fo[cls] = 0;
		}

		//first figure out which neighbor fmessage passes to from current node pass n->? foNode;
		//idea: In bfs traversal order leave to root, check if next the node in bfs order is parent or child of the current node (should be child or parent of the current node)
		int foNode = -1;
		bool foNode_isChild = false;
		for (int p = 0; p < data.allNodes[cur_node_id]->parentsID.size(); p++) {  //check in parent list if found respective parent node is foNode
			int pid = data.allNodes[cur_node_id]->parentsID[p];
			if (!mpVisited[pid]) {
				foNode = pid;
				break;
			}
		}
		if (foNode == -1) {
			for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
				int cid = data.allNodes[cur_node_id]->childrenID[c];
				if (!mpVisited[cid]) {
					foNode = cid;
					foNode_isChild = true;
					break;
				}
			}
		}
		data.allNodes[cur_node_id]->foNode = foNode;
		data.allNodes[cur_node_id]->foNode_ischild = foNode_isChild;
		if (cur_node_id == parameter.bfsRoot && data.allNodes[cur_node_id]->childrenID.size() == 0) {
			foNode_isChild = true;
			data.allNodes[cur_node_id]->foNode_ischild = true;
		}
		//incoming message from visited child
		if (data.allNodes[cur_node_id]->childrenID.size() > 0) {

			for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
				int child_id = data.allNodes[cur_node_id]->childrenID[c];

				if (child_id == foNode) {  //if child_node is foNode skip
										   //for (int c_cls = 0; c_cls < cNum; c_cls++) {
										   //	data.allNodes[cur_node_id]->fi_ChildList[c*cNum + c_cls] = eln(1); //need to confirm
										   //}
					continue;
				}
				//extract parents except current node
				vector<int> parentOfChildExceptCurrentNode;   //Yk E Pc k!=n
				for (int en = 0; en < data.allNodes[child_id]->parentsID.size(); en++) {
					if (data.allNodes[child_id]->parentsID[en] == cur_node_id) { //k!=n
						continue;
					}
					parentOfChildExceptCurrentNode.push_back(data.allNodes[child_id]->parentsID[en]);
				}

				for (int cls = 0; cls < cNum; cls++) {  //cls represents current node class
					double sumAccumulator = eln(0);   //should be 0 since we are summing it up//eln(1);//need to confirm
					for (int c_cls = 0; c_cls < cNum; c_cls++) { //c_cls reperesnets child class label   Yc
						int max_bitCount = 1 << parentOfChildExceptCurrentNode.size();
						for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent and child class label(given by c_cls)
							double productAccumulator = eln(1);
							int parentClsProd = 1; //p(c), product of parent classes for child c

							for (int p = 0; p < parentOfChildExceptCurrentNode.size(); p++) {//calculating Product(fo(p)) for all parent of current child except the current node
								int pid = parentOfChildExceptCurrentNode[p];
								int parentClsValue = (bitCount >> p) & 1;
								parentClsProd *= parentClsValue;
								productAccumulator = elnproduct(productAccumulator, data.allNodes[pid]->fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
							}
							productAccumulator = elnproduct(productAccumulator, data.allNodes[child_id]->fo[c_cls]);  //product with fo(c)
																														 //multiplying P(Yc|Ypc)
							parentClsProd *= cls; //class of current node 
							productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[c_cls][parentClsProd]);
							sumAccumulator = elnsum(sumAccumulator, productAccumulator);
						}
					}
					data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls] = sumAccumulator;
				}
			}
		}

		if (foNode_isChild) {  //means the current node has all visited parents
			if (data.allNodes[cur_node_id]->parentsID.size() == 0) {
				for (int cls = 0; cls < cNum; cls++) {
					data.allNodes[cur_node_id]->fi_parent[cls] = parameter.elnPz[cls];
				}
			}
			else {
				for (int cls = 0; cls < cNum; cls++) {
					double sumAccumulator = eln(0);
					int max_bitCount = 1 << data.allNodes[cur_node_id]->parentsID.size();
					for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent class label
						double productAccumulator = eln(1);
						int parentClsProd = 1;
						for (int p = 0; p < data.allNodes[cur_node_id]->parentsID.size(); p++) {
							int pid = data.allNodes[cur_node_id]->parentsID[p];
							int parentClsValue = (bitCount >> p) & 1;
							parentClsProd *= parentClsValue;
							productAccumulator = elnproduct(productAccumulator, data.allNodes[pid]->fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
						}
						productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[cls][parentClsProd]);
						sumAccumulator = elnsum(sumAccumulator, productAccumulator);
					}
					data.allNodes[cur_node_id]->fi_parent[cls] = sumAccumulator;
				}
			}

			//calulating fo
			for (int cls = 0; cls < cNum; cls++) { //cls represents class of the current node
				double productAccumulator = eln(1);
				for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
					int child_id = data.allNodes[cur_node_id]->childrenID[c];

					if (child_id == foNode) {
						continue;
					}
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls]); //multiplying with al the child fi except the outgoing child
				}
				productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_parent[cls]);  // multiplying with fi(n)_parent
				productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[cur_node_id*cNum + cls]);
				data.allNodes[cur_node_id]->fo[cls] = productAccumulator;
			}

		}

		else {  //message pass n-> parent there is no fi(n)_parent   //computes for root node as well
				//calulating fo
			for (int cls = 0; cls < cNum; cls++) { //cls represents class of the current node
				double productAccumulator = eln(1);
				for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls]); //multiplying with al the child fi except the outgoing child
				}
				productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[cur_node_id*cNum + cls]);
				data.allNodes[cur_node_id]->fo[cls] = productAccumulator;
			}
		}

		mpVisited[cur_node_id] = 1;

		//verification
		for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
			if (data.allNodes[cur_node_id]->childrenID[c] == foNode) {
				for (int cls = 0; cls < cNum; cls++) {
					if (data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls] != 0) {
						cout << " fi_childlist Message Computation Error (this should not be computed)  Node " << cur_node_id << endl;
					}
				}
				continue;
			}
			for (int cls = 0; cls < cNum; cls++) {
				if (data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls] < MESSAGELOW || data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls]>0) {
					cout << " fi_childlist Message Computation Error in Node " << cur_node_id << endl;
				}
			}
		}
		if (foNode_isChild) {
			for (int cls = 0; cls < cNum; cls++) {
				if (data.allNodes[cur_node_id]->fi_parent[cls] < MESSAGELOW || data.allNodes[cur_node_id]->fi_parent[cls]>0) {
					cout << " fi_parent Message Computation Error in Node " << cur_node_id << endl;
				}
			}
		}
		//verify fo 
		for (int cls = 0; cls < cNum; cls++) {
			if (data.allNodes[cur_node_id]->fo[cls] < MESSAGELOW || data.allNodes[cur_node_id]->fo[cls]>0) {
				cout << " fo Message Computation Error in Node " << cur_node_id << endl;
			}
		}

	}


	//root to leaves traversal
	vector<int> gVisited(parameter.allPixelSize, 0);
	//for root node
	//computing gi
	//int root_nodeId = data.bfsTraversalOrder[0]; //root node
	//cout << "parameter.bfsRoot = " << parameter.bfsRoot << endl;
	if (data.allNodes[parameter.bfsRoot]->childrenID.size() == 0) {
		for (int cls = 0; cls < cNum; cls++) {
			data.allNodes[parameter.bfsRoot]->go_parent[cls] = 0;
			data.allNodes[parameter.bfsRoot]->gi[cls] = eln(1);
		}
		data.allNodes[parameter.bfsRoot]->go_fromParent = false;        //case1: if current node has visited parent
		data.allNodes[parameter.bfsRoot]->go_fromChild = true;         //case2: if current node has visited child and if go is in visited child
		data.allNodes[parameter.bfsRoot]->go_fromParentofChild = false; //case3: if current node has visited child and if go is in one of the visited child's parent
		data.allNodes[parameter.bfsRoot]->go_lastVisitedNode = -1;
		for (int cls = 0; cls < cNum; cls++) {
			double productAccumulator = eln(1);
			productAccumulator = elnproduct(productAccumulator, data.allNodes[parameter.bfsRoot]->gi[cls]);
			productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[parameter.bfsRoot*cNum + cls]);
			data.allNodes[parameter.bfsRoot]->go_parent[cls] = productAccumulator;
		}

	}
	else {
		//initializing go_childlist,go_parent gi for root node
		data.allNodes[parameter.bfsRoot]->go_ChildList.resize(data.allNodes[parameter.bfsRoot]->childrenID.size()*cNum, 0);
		for (int cls = 0; cls < cNum; cls++) {
			data.allNodes[parameter.bfsRoot]->go_parent[cls] = 0;
			data.allNodes[parameter.bfsRoot]->gi[cls] = parameter.elnPz[cls];
		}
		data.allNodes[parameter.bfsRoot]->go_fromParent = true;        //case1: if current node has visited parent
		data.allNodes[parameter.bfsRoot]->go_fromChild = false;         //case2: if current node has visited child and if go is in visited child
		data.allNodes[parameter.bfsRoot]->go_fromParentofChild = false; //case3: if current node has visited child and if go is in one of the visited child's parent
		data.allNodes[parameter.bfsRoot]->go_lastVisitedNode = -1;
		//computing go for every child c of n
		for (int c = 0; c < data.allNodes[parameter.bfsRoot]->childrenID.size(); c++) {
			int cid = data.allNodes[parameter.bfsRoot]->childrenID[c];
			for (int cls = 0; cls < cNum; cls++) {
				double productAccumulator = eln(1);
				for (int d = 0; d < data.allNodes[parameter.bfsRoot]->childrenID.size(); d++) {
					if (d == c) continue;
					productAccumulator = elnproduct(productAccumulator, data.allNodes[parameter.bfsRoot]->fi_ChildList[d*cNum + cls]);
				}
				productAccumulator = elnproduct(productAccumulator, data.allNodes[parameter.bfsRoot]->gi[cls]);
				productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[parameter.bfsRoot*cNum + cls]);
				data.allNodes[parameter.bfsRoot]->go_ChildList[c*cNum + cls] = productAccumulator;
			}
		}
	}
	gVisited[parameter.bfsRoot] = 1;
	for (int node = 1; node < data.bfsTraversalOrder.size(); node++) {
		int cur_node_id = data.bfsTraversalOrder[node];  //n
														 //only one gi, many go
														 //initializing go_childlist,go_parent gi
		data.allNodes[cur_node_id]->go_ChildList.resize(data.allNodes[cur_node_id]->childrenID.size()*cNum, 0);
		for (int cls = 0; cls < cNum; cls++) {
			data.allNodes[cur_node_id]->go_parent[cls] = 0;
			data.allNodes[cur_node_id]->gi[cls] = 0;
		}
		//data.allNodes[cur_node_id]->go_ChildList.resize(data.allNodes[cur_node_id]->childrenID.size()*cNum, LOGZERO);

		//first figure out g direction from parent side or      child side (two case: child side or parent of child side)
		data.allNodes[cur_node_id]->go_fromParent = false;        //case1: if current node has visited parent
		data.allNodes[cur_node_id]->go_fromChild = false;         //case2: if current node has visited child and if go is in visited child
		data.allNodes[cur_node_id]->go_fromParentofChild = false; //case3: if current node has visited child and if go is in one of the visited child's parent
		data.allNodes[cur_node_id]->go_lastVisitedNode = -1;



		int visitedCounter = 0;
		for (int p = 0; p < data.allNodes[cur_node_id]->parentsID.size(); p++) {  //check in parent list if found respective parent node is foNode
			int pid = data.allNodes[cur_node_id]->parentsID[p];
			if (gVisited[pid]) {
				data.allNodes[cur_node_id]->go_fromParent = true;
				data.allNodes[cur_node_id]->go_lastVisitedNode = pid;
				visitedCounter++;
				break;
			}
		}
		if (data.allNodes[cur_node_id]->go_lastVisitedNode == -1) {
			for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
				int cid = data.allNodes[cur_node_id]->childrenID[c];
				if (gVisited[cid]) {
					visitedCounter++;
					if (data.allNodes[cid]->go_fromParent) {
						data.allNodes[cur_node_id]->go_fromParentofChild = true;
					}
					else {
						data.allNodes[cur_node_id]->go_fromChild = true;
					}
					data.allNodes[cur_node_id]->go_lastVisitedNode = cid;
					break;
				}
			}
		}
		if (visitedCounter != 1) {
			cout << "Not one visited Neighbour Error" << endl;
		}
		if (data.allNodes[cur_node_id]->go_fromParent == false && data.allNodes[cur_node_id]->go_fromParentofChild == false && data.allNodes[cur_node_id]->go_fromChild == false) {
			cout << "Error all neighbours are not visited Node " << cur_node_id << endl;
		}



		if (data.allNodes[cur_node_id]->go_fromParent) {

			//computing gi
			for (int cls = 0; cls < cNum; cls++) {
				double sumAccumulator = eln(0);
				int max_bitCount = 1 << data.allNodes[cur_node_id]->parentsID.size();
				for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent class label
					double productAccumulator = eln(1);
					int parentClsProd = 1;
					for (int p = 0; p < data.allNodes[cur_node_id]->parentsID.size(); p++) {
						int pid = data.allNodes[cur_node_id]->parentsID[p];
						int parentClsValue = (bitCount >> p) & 1;
						parentClsProd *= parentClsValue;
						//multiply with go(Po)_childlist[n]
						if (pid == data.allNodes[cur_node_id]->go_lastVisitedNode) {
							for (int c = 0; c < data.allNodes[pid]->childrenID.size(); c++) {
								int cid = data.allNodes[pid]->childrenID[c];
								if (cid == cur_node_id) {
									double tempgoChild = data.allNodes[pid]->go_ChildList[c*cNum + parentClsValue];
									productAccumulator = elnproduct(productAccumulator, tempgoChild);
									break;
								}
							}
						}
						else {
							productAccumulator = elnproduct(productAccumulator, data.allNodes[pid]->fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
						}
					}
					productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[cls][parentClsProd]);
					sumAccumulator = elnsum(sumAccumulator, productAccumulator);
				}
				data.allNodes[cur_node_id]->gi[cls] = sumAccumulator;
			}

			//computing go for every child c of n
			for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
				int cid = data.allNodes[cur_node_id]->childrenID[c];
				for (int cls = 0; cls < cNum; cls++) {
					double productAccumulator = eln(1);
					for (int d = 0; d < data.allNodes[cur_node_id]->childrenID.size(); d++) {
						if (d == c) continue;
						productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_ChildList[d*cNum + cls]);
					}
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->gi[cls]);
					productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[cur_node_id*cNum + cls]);
					data.allNodes[cur_node_id]->go_ChildList[c*cNum + cls] = productAccumulator;
				}
			}
		}
		else {

			if (data.allNodes[cur_node_id]->go_fromChild) {
				//computing gi(n)
				int Co = data.allNodes[cur_node_id]->go_lastVisitedNode;
				vector<int> parentOfCoExcept_currentNode;
				for (int en = 0; en < data.allNodes[Co]->parentsID.size(); en++) {
					if (data.allNodes[Co]->parentsID[en] == cur_node_id) {
						continue;
					}
					parentOfCoExcept_currentNode.push_back(data.allNodes[Co]->parentsID[en]);
				}
				for (int cls = 0; cls < cNum; cls++) {  //current node class
					double sumAccumulator = eln(0);
					for (int Co_cls = 0; Co_cls < cNum; Co_cls++) {
						int max_bitCount = 1 << parentOfCoExcept_currentNode.size();
						for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent class label product(fo(p)) except current node
							double productAccumulator = data.allNodes[Co]->go_parent[Co_cls];
							int parentClsProd = 1;
							for (int p = 0; p < parentOfCoExcept_currentNode.size(); p++) {
								int pid = parentOfCoExcept_currentNode[p];
								int parentClsValue = (bitCount >> p) & 1;
								parentClsProd *= parentClsValue;
								productAccumulator = elnproduct(productAccumulator, data.allNodes[pid]->fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
							}
							//p(Yco|Ypco) 
							parentClsProd *= cls;
							productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[Co_cls][parentClsProd]);
							sumAccumulator = elnsum(sumAccumulator, productAccumulator);
						}
					}
					data.allNodes[cur_node_id]->gi[cls] = sumAccumulator;
				}
			}


			else if (data.allNodes[cur_node_id]->go_fromParentofChild) {
				//computing gi(n)
				int Co = data.allNodes[cur_node_id]->go_lastVisitedNode;
				int Po = data.allNodes[Co]->go_lastVisitedNode;
				if (Po == -1) {
					cout << "error: Three should be a parent of a child" << endl;
				}
				int CIndex = -1;
				for (int c = 0; c < data.allNodes[Po]->childrenID.size(); c++) {
					if (data.allNodes[Po]->childrenID[c] == Co) {
						CIndex = c;
						break;
					}
				}
				vector<int> parentOfCoExcept_currentNode;
				for (int en = 0; en < data.allNodes[Co]->parentsID.size(); en++) {
					if (data.allNodes[Co]->parentsID[en] == cur_node_id) {
						continue;
					}
					parentOfCoExcept_currentNode.push_back(data.allNodes[Co]->parentsID[en]);
				}
				for (int cls = 0; cls < cNum; cls++) {  //current node class
					double sumAccumulator = eln(0);
					for (int Co_cls = 0; Co_cls < cNum; Co_cls++) {
						int max_bitCount = 1 << parentOfCoExcept_currentNode.size();
						for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent class label product(fo(p)) except current node
							double productAccumulator = data.allNodes[Co]->fo[Co_cls];
							int parentClsProd = 1;
							for (int p = 0; p < parentOfCoExcept_currentNode.size(); p++) {
								int pid = parentOfCoExcept_currentNode[p];
								int parentClsValue = (bitCount >> p) & 1;
								parentClsProd *= parentClsValue;
								if (pid == Po) {
									double go_Po_child = data.allNodes[Po]->go_ChildList[CIndex*cNum + parentClsValue];
									productAccumulator = elnproduct(productAccumulator, go_Po_child);
								}
								else {
									productAccumulator = elnproduct(productAccumulator, data.allNodes[pid]->fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
								}
							}
							//p(Yco|Ypco) 
							parentClsProd *= cls;
							productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[Co_cls][parentClsProd]);
							sumAccumulator = elnsum(sumAccumulator, productAccumulator);
						}
					}
					data.allNodes[cur_node_id]->gi[cls] = sumAccumulator;
				}
			}

			//computing go(n)_parent
			int Co = data.allNodes[cur_node_id]->go_lastVisitedNode;
			for (int cls = 0; cls < cNum; cls++) {
				double productAccumulator = eln(1);
				for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
					int cid = data.allNodes[cur_node_id]->childrenID[c];
					if (cid == Co) {
						continue;
					}
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_ChildList[c*cNum + cls]);
				}
				productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->gi[cls]);
				productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[cur_node_id*cNum + cls]);
				data.allNodes[cur_node_id]->go_parent[cls] = productAccumulator;
			}

			//computing go(n)_child
			//for every child c of n . c != Co 
			for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
				int cid = data.allNodes[cur_node_id]->childrenID[c];
				if (cid == Co) {
					continue;
				}
				for (int cls = 0; cls < cNum; cls++) {
					double productAccumulator = eln(1);
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_parent[cls]);
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->gi[cls]);
					for (int d = 0; d < data.allNodes[cur_node_id]->childrenID.size(); d++) {
						if (d == c || data.allNodes[cur_node_id]->childrenID[d] == Co) continue;
						productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id]->fi_ChildList[d*cNum + cls]);
					}
					productAccumulator = elnproduct(productAccumulator, parameter.elnPxn_zn[cur_node_id*cNum + cls]);
					data.allNodes[cur_node_id]->go_ChildList[c*cNum + cls] = productAccumulator;
				}
			}
		}
		gVisited[cur_node_id] = 1;
		//verification
		for (int c = 0; c < data.allNodes[cur_node_id]->childrenID.size(); c++) {
			if (data.allNodes[cur_node_id]->childrenID[c] == data.allNodes[cur_node_id]->go_lastVisitedNode) {
				for (int cls = 0; cls < cNum; cls++) {
					if (data.allNodes[cur_node_id]->go_ChildList[c*cNum + cls] != 0) {
						cout << " go_childlist Message Computation Error (this should not be computed)  Node " << cur_node_id << endl;
					}
				}
				continue;
			}
			for (int cls = 0; cls < cNum; cls++) {
				if (data.allNodes[cur_node_id]->go_ChildList[c*cNum + cls] < MESSAGELOW || data.allNodes[cur_node_id]->go_ChildList[c*cNum + cls]>0) {
					cout << " go_childlist Message Computation Error in Node " << cur_node_id << endl;
				}
			}
		}
		if (data.allNodes[cur_node_id]->go_fromChild || data.allNodes[cur_node_id]->go_fromParentofChild) {
			for (int cls = 0; cls < cNum; cls++) {
				if (data.allNodes[cur_node_id]->go_parent[cls] < MESSAGELOW || data.allNodes[cur_node_id]->go_parent[cls]>0) {
					cout << " go_parent Message Computation Error in Node " << cur_node_id << endl;
				}
			}
		}
		//verify gi 
		for (int cls = 0; cls < cNum; cls++) {
			if (data.allNodes[cur_node_id]->gi[cls] < MESSAGELOW || data.allNodes[cur_node_id]->gi[cls]>0) {
				cout << " gi Message Computation Error in Node " << cur_node_id << endl;
			}
		}
	}
}

void cFlood::treeConstrut() {



};

int main(int argc, char *argv[]) {
	cFlood flood;
	flood.input(argc, argv);
}

//struct conMatrix cFlood::getConfusionMatrix() {
//	struct conMatrix confusionMatrix;
//	confusionMatrix.TT = 0;
//	confusionMatrix.TF = 0;
//	confusionMatrix.FF = 0;
//	confusionMatrix.FT = 0;
//
//	for (int i = 0; i < testIndex.size(); i++) {
//		if (testLabel[i] == 0 && mappredictions[testIndex[i]] == 0) {
//			confusionMatrix.FF++;
//		}
//		else if (testLabel[i] == 0 && mappredictions[testIndex[i]] == 1) {
//			confusionMatrix.FT++;
//		}
//		else if (testLabel[i] == 1 && mappredictions[testIndex[i]] == 1) {
//			confusionMatrix.TT++;
//		}
//		else if (testLabel[i] == 1 && mappredictions[testIndex[i]] == 0) {
//			confusionMatrix.TF++;
//		}
//	}
//	return confusionMatrix;
//}



