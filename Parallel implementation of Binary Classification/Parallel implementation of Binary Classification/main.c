#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include "main.h"

int main()
{
	srand(time(0));
	int N = 200;
	int K = 3;
	double dt = 0.10;
	double tmax = 1000;
	double a = 0.5;
	int LIMIT = 200;
	double QC = 0.025;
	int t = 0;

	//-----------------------------------------------------------------------------------------------
	Point* pointArrTest = createPointArr(N, K);
	writePointsToFileTest("ParametersAndPoints.txt", N, K, dt, tmax, a, LIMIT, QC, pointArrTest);
	//-----------------------------------------------------------------------------------------------
	Point* pointArr = readPointsFromFile("ParametersAndPoints.txt", &N, &K, &dt, &tmax, &a, &LIMIT, &QC);
	double* weights = (double*)calloc(K + 1, sizeof(double));
	binaryClassificationAlgorithm(N, K, pointArr, weights, a, LIMIT, QC, dt, tmax);
}



Point* readPointsFromFile(const char* fileName, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC)
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

void binaryClassificationAlgorithm(int N, int K, Point* pointArr, double* weights, double a, int LIMIT, double QC, double dt, double tmax)
{
	double t = 0;
	int Nmiss = checkAllPointsLimitTimes(N, K, pointArr, weights, a, LIMIT);
	printf("Nmiss = %d\n", Nmiss);
	fflush(NULL);
	if (Nmiss == 0) {
		printf("Success! - Zero Nmiss\n");
		fflush(NULL);
	}

	double q = checkQualityOfClassifier(Nmiss, N);

	//-----------------------------------------
	printf("starting q: %lf\n", q);
	//-----------------------------------------

	while (q >= QC)
	{//Not a good Quality
		t = t + dt;
		if (t > tmax)
		{
			printf("Fail! - Time passed\n");
			fflush(NULL);
			break;
		}
		updateAllPointsCoordinantes(pointArr, t, K, N);
		Nmiss = checkAllPointsLimitTimes(N, K, pointArr, weights, a, LIMIT);
		q = checkQualityOfClassifier(Nmiss, N);
	}

	//------------------
	if (q < QC && Nmiss != 0)
	{
		printf("Half Success! - Quality check was OK but Nmiss was NOT zero\n");
		fflush(NULL);
	}
	//------------------

	printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
	printf("%d\t%d\t%lf\t%lf\t%lf\t%d\t%lf\n", N, K, dt, tmax, a, LIMIT, QC);
	printf("Alpha minimum = %lf     q = %lf", a, q);
	printPointArr(pointArr, N, K);
	printWeights(weights, K);
	printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");


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

double checkQualityOfClassifier(int Nmiss, int N)
{
	return ((double)(Nmiss)) / ((double)N);
}

int checkAllPointsLimitTimes(int N, int K, Point* pointArr, double* weights, double a, int LIMIT)
{
	int i;
	int Nmiss;
	checkAllPointsOneIteration(N, K, pointArr, weights, a, &Nmiss);
	for (i = 0; (i < LIMIT - 1) && (Nmiss != 0); i++)
	{
		Nmiss = 0;
		checkAllPointsOneIteration(N, K, pointArr, weights, a, &Nmiss);
	}

	return Nmiss;
}

void checkAllPointsOneIteration(int N, int K, Point* pointArr, double* weights, double a, int* Nmiss)
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

void printWeights(double* weights, int K)
{
	int i;
	printf("The weights:\n");
	for (i = 0; i < K + 1; i++)
	{
		printf("%lf\n", weights[i]);
	}
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

void printPointArr(Point* pointArr, int N, int K)
{
	int i;
	printf("\n-----------Points Array-------------\n");
	for (i = 0; i < N; i++) {
		printf("Point Index %d:\t", i);
		printOnePoint(&(pointArr[i]), K);
		printf("\n");
	}
	printf("------------------------------------\n");
}