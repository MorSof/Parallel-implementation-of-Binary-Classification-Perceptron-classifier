#pragma once
#pragma once

typedef struct {

	double* coordinantes;
	double* velocity;
	int group;

}Point;


Point* readPointsFromFile(const char* fileName, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId);
Point readOnePointFromFile(int K, FILE* fp);

void binaryClassificationAlgorithm(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, double QC, double dt, double tmax);
void updateAllPointsCoordinantes(Point* pointsArr, double t, int K, int N);
void updateOnePointCoordinantes(Point* pPoint, double t, int K);
double checkQualityOfClassifier(int Nmiss, int N);
int checkAllPointsLimitTimes(int N, int K, Point* pointArr, double* weights, double a, int LIMIT);
void checkAllPointsOneIteration(int N, int K, Point* pointArr, double* weights, double a, int* Nmiss);
void fixWeights(int K, int s, Point* pPoint, double* weights, double a);
int calculateWeightFunc(Point* pPoint, double* weights, int K);
void printWeights(double* weights, int K);


//------------------------------------------------------------
void writePointsToFileTest(const char* fileName, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, Point* pointsArr);
void writeOnePointToFile(FILE* fp, Point* pPoint, int K);
Point createOnePoint(int K);
Point* createPointArr(int N, int K);
void printOnePoint(Point* pPoint, int K);
void printPointArr(Point* pointArr, int N, int K, int myId);
