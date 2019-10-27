#pragma once

#define MASTER 0
#define POINT_NUM_OF_ATTRIBUTES 3
#define RESULT_TAG 0
#define TERMINATION_TAG 1
#define WEIGHTS_TAG 2
#define GLOBAL_SUCCESS_TAG 3
#define GLOBAL_FAIL_TAG 4
#define CONTINUATION_TAG 5
#define FAIL -1
#define SUCCESS 1
#define CONTINIUE 1

typedef struct {

	double* coordinantes;
	double* velocity;
	int group;

}Point;


Point* readPointsFromFile(const char* fileName, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId);
Point readOnePointFromFile(int K, FILE* fp);

int searchForFirstSuccess(double* doneJobsPerRound, int numOfWorkingProccesses);
int binaryClassificationAlgorithm(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, double* q, double QC, int myId);
//void binaryClassificationAlgorithm(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, double QC, double dt, double tmax, int myId);
void updateAllPointsCoordinantes(Point* pointsArr, double t, int K, int N);
void updateOnePointCoordinantes(Point* pPoint, double t, int K);
double checkQualityOfClassifier(int Nmiss, int N);
int checkAllPointsLimitTimes(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, int myId);
void checkAllPointsOneIteration(int N, int K, Point* pointArr, double* weights, double a, int* Nmiss, int myId);
void initWeights(double* weights, int K);
void fixWeights(int K, int sign, Point* pPoint, double* weights, double a);
//void fixWeights(int K, int sign, double* coordinantes, double* weights, double a);
int calculateWeightFunc(Point* pPoint, double* weights, int K);
void printWeights(double* weights, int K, int myId);


//------------------------------------------------------------
void writePointsToFileTest(const char* fileName, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, Point* pointsArr);
void writeOnePointToFile(FILE* fp, Point* pPoint, int K);
Point createOnePoint(int K);
Point* createPointArr(int N, int K);
void printOnePoint(Point* pPoint, int K);
void printPointArr(Point* pointArr, int N, int K, int myId);
void printDoneJobs(double* doneJobsPerRound, int numOfWorkingSlaves);
