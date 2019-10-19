#pragma once
#include <mpi.h>
#include "main.h"

void scatterPacks(Point* pointArr, MPI_Datatype PointType, int K, int numOfSlaves, int numOfPointsRemainder, int numOfPointsPerSlave);
void createPointType(MPI_Datatype* PointType);
char* creatPack(Point* pointArr, MPI_Datatype PointType, int* position, int N, int K);
Point* recievePack(MPI_Datatype PointType, int myId, int K, int numOfPointsToRecieve);
Point* openPack(MPI_Datatype PointType, char* buffer, int bufferSize, int N, int K);
void broadcastParameters(int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC);

