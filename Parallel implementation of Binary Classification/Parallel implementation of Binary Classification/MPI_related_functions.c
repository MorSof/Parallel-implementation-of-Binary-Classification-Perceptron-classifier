#define _CRT_SECURE_NO_WARNINGS


#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"
#include "MPI_related_functions.h"


void sendMessageToAllSlaves(int numOfWorkingSlaves, int message, int tag)
{
	int i, slaveId;
	for (i = 0; i < numOfWorkingSlaves; i++)
	{
		slaveId = i + 1;
		MPI_Send(&message, 1, MPI_INT, slaveId, tag, MPI_COMM_WORLD);
	}
}


void broadcastParameters(int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC)
{
	/*MPI_Bcast(
		void* data,
		int count,
		MPI_Datatype datatype,
		int root,
		MPI_Comm communicator)*/

	MPI_Bcast(N, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(dt, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(tmax, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(a, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(LIMIT, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(QC, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);


}

void broadcastPacks(Point* pointArr, MPI_Datatype PointType, int N, int K, int 	numOfWorkingSlaves)
{
	int i;
	int position = 0;
	int slaveId;
	char* buffer = creatPack(pointArr, PointType, &position, N, K);
	for (i = 0; i < numOfWorkingSlaves; i++)
	{
		slaveId = i + 1;		
		MPI_Send(buffer, position, MPI_PACKED, slaveId, 0, MPI_COMM_WORLD);
	}
	free(buffer);
}

void scatterPacks(Point* pointArr, MPI_Datatype PointType, int K, int numOfSlaves, int numOfPointsRemainder, int numOfPointsPerSlave)
{
	int i;
	int position;
	int slaveId;
	int numOfPointsToSend;
	Point* startPointArr;

	for (i = 0; i < numOfSlaves; i++)
	{
		position = 0;
		slaveId = i + 1;
		numOfPointsToSend = slaveId <= numOfPointsRemainder ? numOfPointsPerSlave + 1 : numOfPointsPerSlave; // spliting the remainder equally
		startPointArr = pointArr + (numOfPointsToSend * i);
		char* buffer = creatPack(startPointArr, PointType, &position, numOfPointsToSend, K);
		MPI_Send(buffer, position, MPI_PACKED, slaveId, 0, MPI_COMM_WORLD);
		free(buffer);
	}
}

char* creatPack(Point* pointArr, MPI_Datatype PointType, int* position, int N, int K)
{
	int i;
	int bufferSize = N * (sizeof(Point) + K * sizeof(double) + K * sizeof(double));
	char* buffer = (char*)malloc(bufferSize);
	MPI_Pack(pointArr, N, PointType, buffer, bufferSize, position, MPI_COMM_WORLD);
	for (i = 0; i < N; i++)
	{
		MPI_Pack(pointArr[i].coordinantes, K, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
		MPI_Pack(pointArr[i].velocity, K, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	}
	return buffer;
}

Point* recievePack(MPI_Datatype PointType, int myId, int K, int N)
{
	MPI_Status status;
	int position = 0;
	int bufferSize;
	char* buffer;
	bufferSize = N * (sizeof(Point) + K * sizeof(double) + K * sizeof(double));
	buffer = (char*)malloc(bufferSize);
	MPI_Recv(buffer, bufferSize, MPI_PACKED, MASTER, 0, MPI_COMM_WORLD, &status);
	Point* pointArr = openPack(PointType, buffer, bufferSize, N, K);
	return pointArr;
}

//Point* recievePack(MPI_Datatype PointType, int myId, int K, int numOfPointsToRecieve)
//{
//	MPI_Status status;
//	int position = 0;
//	int bufferSize;
//	char* buffer;
//	bufferSize = numOfPointsToRecieve * (sizeof(Point) + K * sizeof(double) + K * sizeof(double));
//	buffer = (char*)malloc(bufferSize);
//	MPI_Recv(buffer, bufferSize, MPI_PACKED, MASTER, 0, MPI_COMM_WORLD, &status);
//	Point* pointArr = openPack(PointType, buffer, bufferSize, numOfPointsToRecieve, K);
//	return pointArr;
//}

Point* openPack(MPI_Datatype PointType, char* buffer, int bufferSize, int N, int K)
{
	int i;
	int position = 0;
	Point* pointArr = (Point*)malloc(N * sizeof(Point));
	MPI_Unpack(buffer, bufferSize, &position, pointArr, N, PointType, MPI_COMM_WORLD);
	for (i = 0; i < N; i++)
	{
		pointArr[i].coordinantes = (double*)malloc(sizeof(double) * K);
		pointArr[i].velocity = (double*)malloc(sizeof(double) * K);

		MPI_Unpack(buffer, bufferSize, &position, pointArr[i].coordinantes, K, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Unpack(buffer, bufferSize, &position, pointArr[i].velocity, K, MPI_DOUBLE, MPI_COMM_WORLD);
	}

	return pointArr;
}

void createPointType(MPI_Datatype* PointType)
{
	int blockLengths[POINT_NUM_OF_ATTRIBUTES] = { 1, 1, 1 };
	MPI_Aint disp[POINT_NUM_OF_ATTRIBUTES];
	MPI_Datatype types[POINT_NUM_OF_ATTRIBUTES] = { MPI_DOUBLE, MPI_DOUBLE ,MPI_INT };

	disp[0] = offsetof(Point, coordinantes);
	disp[1] = offsetof(Point, velocity);
	disp[2] = offsetof(Point, group);

	MPI_Type_create_struct(POINT_NUM_OF_ATTRIBUTES, blockLengths, disp, types, PointType);
	MPI_Type_commit(PointType);
}
