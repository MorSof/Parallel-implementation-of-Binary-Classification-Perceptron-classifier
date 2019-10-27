
#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "main.h"
#include "MPI_related_functions.h"



	/*MPI_Send(
			void* data,
			int count,
			MPI_Datatype datatype,
			int destination,
			int tag,
			MPI_Comm communicator)*/


	/*MPI_Recv(
			void* data,
			int count,
			MPI_Datatype datatype,
			int source,
			int tag,
			MPI_Comm communicator,
			MPI_Status* status)*/



int main(int argc, char *argv[])
{
	int myId, numOfProcess;
	MPI_Init(&argc, &argv);
	MPI_Datatype PointType;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
	int numOfSlaves = numOfProcess - 1;

	srand(time(0));
	int N = 100000;
	int K = 2;
	double dt = 4;
	double tmax = 400;
	double a = 0.5;
	int LIMIT = 200;
	double QC = 0.5;
	double t = 0;
	int numOfJobs = (int)(tmax / dt) + 1;
	int numOfWorkingProccesses = numOfProcess > numOfJobs ? numOfJobs : numOfProcess; //if there are more slaves then jobs => the unimployed slaves are not relevant
	int numOfWorkingSlaves = numOfSlaves > numOfJobs ? numOfJobs : numOfSlaves;
	double* weights = (double*)calloc(K + 1, sizeof(double));

	createPointType(&PointType);
	//-----------------------------------------------------------------------------------------------
	/*if (myId == MASTER) 
	{
		Point* pointArrTest = createPointArr(N, K);
		writePointsToFileTest("C:\\Users\\morso\\source\\repos\\Perceptron-classifier\\Parallel implementation of Binary Classification\\Parallel implementation of Binary Classification\\Points.txt", N, K, dt, tmax, a, LIMIT, QC, pointArrTest);
	}*/
	//-----------------------------------------------------------------------------------------------
	
	clock_t c = clock();
	
	Point* pointArr = NULL;
	if (myId == MASTER)
	{
		pointArr = readPointsFromFile("C:\\Users\\morso\\source\\repos\\Perceptron-classifier\\Parallel implementation of Binary Classification\\Parallel implementation of Binary Classification\\Points.txt", &N, &K, &dt, &tmax, &a, &LIMIT, &QC, myId);
	}
	
	broadcastParameters(&N, &K, &dt, &tmax, &a, &LIMIT, &QC);

	if (myId == MASTER)
	{
		
		double* doneJobsPerRound = NULL; //if the job will successied it will poccess q in its index, else it will poccess -1
		int winnerSlavesId, searchResult;
		int localResult;
		int round = 0;
		int i;
		double tPivotPerRound = 0;
		broadcastPacks(pointArr, PointType, N, K, numOfWorkingSlaves);
		doneJobsPerRound = (double*)malloc(numOfWorkingSlaves * sizeof(double));


		/*printf("\n------------------MASTER: Starting DATA--------------------\n");
		fflush(NULL);
		printPointArr(pointArr, N, K, myId);
		fflush(NULL);
		printf("numOfWorkingSlaves: %d\n", numOfWorkingSlaves);
		fflush(NULL);
		printf("numOfJobs: %d", numOfJobs);
		fflush(NULL);
		printf("\n------------------------------------------------------------\n");
		fflush(NULL);*/

		while (numOfJobs > 0)
		{
			/*printf("\n------------------MASTER: Round number %d--------------------\n",round);
			fflush(NULL);*/
			numOfWorkingSlaves = numOfSlaves > numOfJobs ? numOfJobs : numOfSlaves; //if there are more slaves then jobs => the unimployed slaves are not relevant			
			for (i = 0; i < numOfWorkingSlaves; i++)
			{
				MPI_Recv(&localResult, 1, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
				doneJobsPerRound[status.MPI_SOURCE - 1] = localResult;
				numOfJobs--;
			}
			//printDoneJobs(doneJobsPerRound, numOfWorkingSlaves);
			searchResult = searchForFirstSuccess(doneJobsPerRound, numOfWorkingSlaves); //if it find a success it return its index, else return FAIL
			if (searchResult != FAIL)
			{
				winnerSlavesId = searchResult + 1;
				sendMessageToAllSlaves(numOfWorkingSlaves, winnerSlavesId, GLOBAL_SUCCESS_TAG);
				break;
			}
			else if (numOfJobs == 0)				
			{
				sendMessageToAllSlaves(numOfWorkingSlaves, FAIL, GLOBAL_FAIL_TAG);
			}
			else
			{
				sendMessageToAllSlaves(numOfWorkingSlaves, FAIL, CONTINUATION_TAG);
				round++;
			}
		/*	printf("\n------------------------------------------------------\n");
			fflush(NULL);*/
		}

		c = clock() - c;
		double time_taken = ((double)c) / CLOCKS_PER_SEC; // calculate the elapsed time
		printf("\nThe program took %f seconds to execute\n", time_taken);

	}
	else //Slaves
	{
		
		double qLocal;
		int message, result;
		int myCurrentJobIndex = myId - 1;
		if (myCurrentJobIndex >= numOfJobs)
		{
			/*printf("myId %d exit the program\n", myId);
			fflush(NULL);*/
			MPI_Finalize();
			return 0;
		}
		pointArr = recievePack(PointType, myId, K, N);
		
		while (myCurrentJobIndex < numOfJobs)
		{
			numOfWorkingSlaves = numOfSlaves > numOfJobs ? numOfJobs : numOfSlaves;
			t = t + ((double)myCurrentJobIndex) * dt;
			updateAllPointsCoordinantes(pointArr, t, K, N);
			initWeights(weights, K);
			result = binaryClassificationAlgorithm(N, K, pointArr, weights, a, LIMIT, &qLocal,QC, myId);
			/*printWeights(weights, K, myId);
			fflush(NULL);*/
			MPI_Send(&result, 1, MPI_INT, MASTER, RESULT_TAG, MPI_COMM_WORLD);
			MPI_Recv(&message, 1, MPI_INT,MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);			
			if (status.MPI_TAG == GLOBAL_SUCCESS_TAG)
			{
				if (myId == message)
				{
					printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
					printf("SUCCESS\n");
					printf("%d\t%d\t%lf\t%lf\t%lf\t%d\t%lf\n", N, K, dt, tmax, a, LIMIT, QC);
					printf("Alpha minimum = %lf     q = %lf\n", a, qLocal);
					//printPointArr(pointArr, N, K, myId);
					printWeights(weights, K, myId);
					printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
					//fflush(NULL);
				}
				break;
			}		
			myCurrentJobIndex += numOfWorkingSlaves;
		}

		if (status.MPI_TAG == GLOBAL_FAIL_TAG && myId == numOfWorkingSlaves)
		{
			printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
			printf("FAIL\n");
			printf("%d\t%d\t%lf\t%lf\t%lf\t%d\t%lf\n", N, K, dt, tmax, a, LIMIT, QC);
			printf("Alpha minimum = %lf     q = %lf\n", a, qLocal);
			//printPointArr(pointArr, N, K, myId);
			printWeights(weights, K, myId);
			printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
			//fflush(NULL);
		}
	}

	/*printf("myId %d exit the program\n", myId);
	fflush(NULL);*/
	MPI_Finalize();
}



int binaryClassificationAlgorithm(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, double* q, double QC,int myId)
{
	int Nmiss = checkAllPointsLimitTimes(N, K, pointArr, weights, a, LIMIT, myId);
	*q = checkQualityOfClassifier(Nmiss, N);
	if (*q < QC) 
	{
		/*printf("myId %d: Success!  q < QC !\n", myId);
		fflush(NULL);*/
		return SUCCESS;
		
	}
		/*printf("myId %d: Fail! - %d Nmiss\n", myId, Nmiss);
		fflush(NULL);*/
		return FAIL;

}

double checkQualityOfClassifier(int Nmiss, int N)
{
	return ((double)(Nmiss)) / ((double)N);
}

int checkAllPointsLimitTimes(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, int myId)
{
	int i;
	int Nmiss;
	checkAllPointsOneIteration(N, K, pointArr, weights, a, &Nmiss, myId);
	for (i = 0; (i < LIMIT - 1) && (Nmiss != 0); i++)
	{
		Nmiss = 0;
		checkAllPointsOneIteration(N, K, pointArr, weights, a, &Nmiss, myId);
	}

	return Nmiss;
}

void checkAllPointsOneIteration(int N, int K, Point* pointArr, double* weights, double a, int* Nmiss, int myId)
{
	int i;

	for (i = 0; i < N; i++)
	{
		/*printPointArr(pointArr, N, K);
		printWeights(weights, K);*/
		int sign = calculateWeightFunc(&(pointArr[i]), weights, K);
		if ((sign == -1 && pointArr[i].group == 1) || (sign == 1 && pointArr[i].group == -1))
		{
			(*Nmiss)++;
			//printf("Point in index %d need to bi fix\n", i);
			fixWeights(K, sign, &pointArr[i], weights, a);
		}
	}
	//-------------------------------
	if (Nmiss == 0) {
		printf("Success!\n");
	}
	//-------------------------------
}

void fixWeights(int K, int sign, Point* pPoint, double* weights, double a)
{
	//W = W + [a*sign(f(P))]P
	int i;
	const double FIRST_PARAM = 1;
	weights[0] = weights[0] + a * ((double)(-1 * sign))*FIRST_PARAM;
	for (i = 0; i < K; i++)
	{
		weights[i + 1] = weights[i + 1] + ((a * ((double)(-1 * sign)))*(pPoint->coordinantes[i]));
	}

}

int calculateWeightFunc(Point* pPoint, double* weights, int K)
{
	double sum = 0;
	int i;
	const double FIRST_PARAM = 1;
	sum += FIRST_PARAM * weights[0];
	for (i = 0; i < K; i++)
	{
		sum += pPoint->coordinantes[i] * weights[i + 1];
	}

	return sum < 0 ? -1 : 1;
}


void updateAllPointsCoordinantes(Point* pointsArr, double t, int K, int N)
{
	int i;
	for (i = 0; i < N; i++)
	{
		updateOnePointCoordinantes(&pointsArr[i], t, K);
	}
}

void updateOnePointCoordinantes(Point* pPoint, double t, int K)
{
	int i;
	for (i = 0; i < K; i++)
	{	// P = P(0) + V*t ;
		pPoint->coordinantes[i] = pPoint->coordinantes[i] + pPoint->velocity[i] * t;
	}
}

int searchForFirstSuccess(double* doneJobsPerRound, int numOfWorkingProccesses)
{
	int i;
	for (i = 0; i < numOfWorkingProccesses; i++)
	{
		if (doneJobsPerRound[i] >= 0)
		{
			return i;
		}
	}
	return -1;
}

Point* readPointsFromFile(const char* fileName, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId)
{
	Point* pointsArr;
	FILE* fp;
	int i;
	fp = fopen(fileName, "r");
	if (!fp)
	{
		return NULL;
	}

	fscanf(fp, "%d", N);
	fscanf(fp, "%d", K);
	fscanf(fp, "%lf", dt);
	fscanf(fp, "%lf", tmax);
	fscanf(fp, "%lf", a);
	fscanf(fp, "%d", LIMIT);
	fscanf(fp, "%lf", QC);

	pointsArr = (Point*)malloc((*N) * sizeof(Point));
	if (!pointsArr)
	{
		fclose(fp);
		return NULL;
	}

	for (i = 0; i < (*N); i++)
	{
		pointsArr[i] = readOnePointFromFile((*K), fp);
	}

	fclose(fp);
	return pointsArr;
}

Point readOnePointFromFile(int K, FILE* fp)
{
	int i;
	Point point;
	point.coordinantes = (double*)malloc(K * sizeof(double));
	point.velocity = (double*)malloc(K * sizeof(double));

	for (i = 0; i < K; i++)
	{
		fscanf(fp, "%lf", &(point.coordinantes[i]));
	}
	for (i = 0; i < K; i++)
	{
		fscanf(fp, "%lf", &(point.velocity[i]));
	}
	fscanf(fp, "%d", &(point.group));
	return point;
}

void writePointsToFileTest(const char* fileName, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, Point* pointsArr)
{
	int i;
	FILE *fp;
	fp = fopen(fileName, "wt");
	if (!fp)
	{
		printf("Cannot Open File '%s'", fileName);
		return;
	}

	fprintf(fp, "%d\t", N);
	fprintf(fp, "%d\t", K);
	fprintf(fp, "%lf\t", dt);
	fprintf(fp, "%lf\t", tmax);
	fprintf(fp, "%lf\t", a);
	fprintf(fp, "%d\t", LIMIT);
	fprintf(fp, "%lf\n", QC);

	for (i = 0; i < N; i++)
	{
		writeOnePointToFile(fp, &(pointsArr[i]), K);
	}

	fclose(fp);

}

void writeOnePointToFile(FILE* fp, Point* pPoint, int K)
{
	int i;
	for (i = 0; i < K; i++)
	{
		fprintf(fp, "%lf\t", pPoint->coordinantes[i]);
	}
	for (i = 0; i < K; i++)
	{
		fprintf(fp, "%lf\t", pPoint->velocity[i]);
	}

	fprintf(fp, " %d\n", pPoint->group);
}

void initWeights(double* weights, int K)
{
	int i;
	for (i = 0; i < K + 1; i++)
	{
		weights[i] = 0;
	}
}

void printWeights(double* weights, int K, int myId)
{
	int i;

	printf("------The weights:   Id %d--------\n", myId);
	for (i = 0; i < K + 1; i++)
	{
		printf("%lf\n", weights[i]);
	}
	fflush(NULL);
}


Point* createPointArr(int N, int K)
{

	int i;
	Point* pointsArr = (Point*)malloc(N * sizeof(Point));
	for (i = 0; i < N; i++)
	{
		pointsArr[i] = createOnePoint(K);
	}

	return pointsArr;
}

Point createOnePoint(int K)
{
	int lower = -10;
	int upper = 10;
	int i;
	Point point;
	point.coordinantes = (double*)malloc(K * sizeof(double));
	point.velocity = (double*)malloc(K * sizeof(double));


	for (i = 0; i < K; i++)
	{
		point.coordinantes[i] = (rand() % (upper - lower + 1)) + lower;
	}
	for (i = 0; i < K; i++)
	{
		point.velocity[i] = (rand() % (upper - lower + 1)) + lower;
	}

	upper = 1;
	lower = -1;
	do {
		point.group = (rand() % (upper - lower + 1)) + lower;
	} while (point.group == 0);

	return point;
}

void printOnePoint(Point* pPoint, int K)
{
	int i;

	for (i = 0; i < K; i++)
	{
		printf("%lf\t", pPoint->coordinantes[i]);
	}
	for (i = 0; i < K; i++)
	{
		printf("%lf\t", pPoint->velocity[i]);
	}

	printf("%d ", pPoint->group);
}

void printPointArr(Point* pointArr, int N, int K, int myId)
{
	int i;
	printf("\n-----------Points Array id %d-------------\n", myId);
	for (i = 0; i < N; i++) {
		printf("Point Index %d:\t", i);
		printOnePoint(&(pointArr[i]), K);
		printf("\n");
	}
	printf("------------------------------------\n");
}

void printDoneJobs(double* doneJobsPerRound, int numOfWorkingSlaves)
{
	int i;
	printf("doneJobsPerRound: \n");
	for ( i = 0; i < numOfWorkingSlaves; i++)
	{
		printf("%lf\t", doneJobsPerRound[i]);
		fflush(NULL);
	}
	printf("\n");
}

//int numOfpointsRemainder = N % numOfSlaves;
	//int numOfpointsPerSlave = N / numOfSlaves;
	//double* weights = (double*)calloc(K + 1, sizeof(double));

	//if (myId == MASTER)
	//{
	//	int tag, globalNmiss = 0;
	//	int numOfSuccess = 0;
	//	int counter = 0;
	//	double q;
	//	double* coordinantes = (double*)malloc(K * sizeof(double));

	//	scatterPacks(pointArr, PointType, K, numOfSlaves, numOfpointsRemainder, numOfpointsPerSlave);
	//	
	//	do {
	//		printf("\n\n********************************************PHASE %d***********************************************\n", counter);
	//		fflush(NULL);
	//		printWeights(weights, K, myId);
	//		sendWeightsToAllSlaves(weights, numOfSlaves, K, DATA_TAG, myId);
	//		tag = recieveCoordsFromAllSlaves(weights, coordinantes, K, a, numOfSlaves, myId);
	//		if (counter == LIMIT - 1)
	//		{				
	//			globalNmiss = recieveNmissFromAllSlaves(numOfSlaves);
	//			if (globalNmiss == -1)
	//			{
	//				printf("Master: TimeOut Fail!\n");
	//				fflush(NULL);
	//				break;
	//			}
	//			q = checkQualityOfClassifier(globalNmiss, N);
	//			if (q < QC)
	//			{
	//				sendWeightsToAllSlaves(weights, numOfSlaves, K, GOOD_QUALITY_TAG, myId);
	//				printf("Master: Good Quality!\n");
	//				fflush(NULL);
	//				break;
	//			}
	//			counter = -1;
	//		}
	//		counter++;
	//		printf("\n*****************************************************************************************************\n");
	//		fflush(NULL);
	//	} while (tag != GLOBAL_SUCCESS_TAG && tag != TERMINATION_TAG);

	//	if (tag == GLOBAL_SUCCESS_TAG)
	//	{
	//		sendWeightsToAllSlaves(weights, numOfSlaves, K, GLOBAL_SUCCESS_TAG, myId);
	//		printf("Master: Global Success!\n");
	//		fflush(NULL);
	//	}
	//	else
	//	{
	//		printf("Master: Slave terminated me!\n");
	//		fflush(NULL);

	//	}

	//	sendWeightsToAllSlaves(weights, numOfSlaves, K, tag, myId);
	//}
	//else
	//{
	//	int numOfPointsToRecieve = myId <= numOfpointsRemainder ? numOfpointsPerSlave + 1 : numOfpointsPerSlave; // spliting the remainder equally
	//	pointArr = recievePack(PointType, myId, K, numOfPointsToRecieve);
	//	binaryClassificationAlgorithm(numOfPointsToRecieve, K, pointArr, weights, a, LIMIT, QC, dt, tmax , myId);
	//}
//}


//void binaryClassificationAlgorithm(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, double QC, double dt, double tmax, int myId)
//{
//	double t = 0;
//	int localNmiss;
//
//	while (t <= tmax)
//	{	
//		printPointArr(pointArr, N, K, myId);
//		fflush(NULL);
//		localNmiss = checkAllPointsLimitTimes(N, K, pointArr, weights, a, LIMIT, myId);
//		if (localNmiss == -1)
//		{// It means that i recieve from master that q < QC
//			return;
//		}
//		MPI_Send(&localNmiss, 1, MPI_INT, MASTER, DATA_TAG, MPI_COMM_WORLD);
//		t = t + dt;
//		updateAllPointsCoordinantes(pointArr, t, K, N);
//	}
//	
//		//TimeOut!
//		MPI_Send(&localNmiss, 1, MPI_INT, MASTER, TIMEOUT_TAG, MPI_COMM_WORLD);
//		printf("myId %d: Fail!\n", myId);
//		fflush(NULL);
//	
//	//printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
//	//printf("%d\t%d\t%lf\t%lf\t%lf\t%d\t%lf\n", N, K, dt, tmax, a, LIMIT, QC);
//	//printf("Alpha minimum = %lf     q = %lf", a, q);
//	///*printPointArr(pointArr, N, K);*/
//	//printWeights(weights, K, myId);
//	//printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
//}
//
//int checkAllPointsLimitTimes(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, int myId)
//{
//	MPI_Status status;
//	int i;
//	int Nmiss = 0;
//
//	
//
//	for (i = 0; (i < LIMIT); i++)
//	{
//		/*printf("myId is: %d\n", myId);
//		fflush(NULL);*/
//		Nmiss = 0;
//		MPI_Recv(weights, K + 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
//		
//
//		if (status.MPI_TAG == GOOD_QUALITY_TAG || status.MPI_TAG == GLOBAL_SUCCESS_TAG)
//		{
//			printf("myId %d: I recieve from master that q < QC Or a Global Success\n", myId);
//			fflush(NULL);
//			return -1;
//		}
//
//		printWeights(weights, K, myId);
//
//		checkAllPointsOneIteration(N, K, pointArr, weights, a, &Nmiss, myId);
//		if (Nmiss == 0)
//		{
//			MPI_Send(pointArr[0].coordinantes, K, MPI_DOUBLE, MASTER, LOCAL_SUCCESS_TAG, MPI_COMM_WORLD);
//			printf("myId %d Local Success!\n", myId);
//			fflush(NULL);
//		}
//	}
//
//	return Nmiss;
//}

//void checkAllPointsOneIteration(int N, int K, Point* pointArr, double* weights, double a, int* Nmiss, int myId)
//{
//	int i, tag, sign;
//	
//	for (i = 0; i < N; i++)
//	{
//		/*printPointArr(pointArr, N, K);
//		printWeights(weights, K);*/
//		sign = calculateWeightFunc(&(pointArr[i]), weights, K);
//		if ((sign == -1 && pointArr[i].group == 1) || (sign == 1 && pointArr[i].group == -1))
//		{
//			(*Nmiss)++;
//			tag = sign >= 0 ? SIGN_POSITIVE_TAG : SIGN_NEGATIVE_TAG;
//			MPI_Send(pointArr[i].coordinantes, K, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
//			return;
//		}
//	}
//
//}
//
//
//double checkQualityOfClassifier(int Nmiss, int N)
//{
//	return ((double)(Nmiss)) / ((double)N);
//}
//
//
//void fixWeights(int K, int sign, double* coordinantes, double* weights, double a)
//{
//	//W = W + [a*sign(f(P))]P
//	int i;
//	const double FIRST_PARAM = 1;
//	weights[0] = weights[0] + a * ((double)(-1 * sign))*FIRST_PARAM;
//	for (i = 0; i < K; i++)
//	{
//		weights[i + 1] = weights[i + 1] + ((a * ((double)(-1 * sign)))*(coordinantes[i]));
//	}
//
//}
//
//int calculateWeightFunc(Point* pPoint, double* weights, int K)
//{
//	double sum = 0;
//	int i;
//	const double FIRST_PARAM = 1;
//	sum += FIRST_PARAM * weights[0];
//	for (i = 0; i < K; i++)
//	{
//		sum += pPoint->coordinantes[i] * weights[i + 1];
//	}
//
//	return sum < 0 ? -1 : 1;
//}

