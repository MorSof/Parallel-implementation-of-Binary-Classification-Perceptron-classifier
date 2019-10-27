#pragma once
#include <mpi.h>
#include "main.h"

void sendMessageToAllSlaves(int numOfWorkingSlaves, int message, int tag);
//int recieveCoordsFromAllSlaves(double* weights, double* coordinantes, int K, double a, int numOfSlaves, int myId);
//void sendWeightsToAllSlaves(double* weights, int numOfSlaves, int K, int tag, int myId);
//int recieveNmissFromAllSlaves(int numOfSlaves);
void broadcastParameters(int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC);
void broadcastPacks(Point* pointArr, MPI_Datatype PointType, int N, int K, int 	numOfWorkingSlaves);
void scatterPacks(Point* pointArr, MPI_Datatype PointType, int K, int numOfSlaves, int numOfPointsRemainder, int numOfPointsPerSlave);
void createPointType(MPI_Datatype* PointType);
char* creatPack(Point* pointArr, MPI_Datatype PointType, int* position, int N, int K);
Point* recievePack(MPI_Datatype PointType, int myId, int K, int N);
//Point* recievePack(MPI_Datatype PointType, int myId, int K, int numOfPointsToRecieve);
Point* openPack(MPI_Datatype PointType, char* buffer, int bufferSize, int N, int K);
void broadcastParameters(int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC);

