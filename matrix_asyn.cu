#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
using namespace std;

__global__ void sKNN(float *attr, int *val, int n_att, int n_inst, float *distance, int com, float *smallestDistance, int stream, int elementsstream){

	//calculate tid

	int column = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
	int idrow = stream * elementsstream + row;
	int tid = (blockDim.x*gridDim.x * idrow)+column;

	if (tid < n_inst * com){
		smallestDistance[tid] = FLT_MAX;
	}
	if (column < n_inst && row < elementsstream  && idrow < n_inst){


			float diff;

			distance[idrow * n_inst + column] = 0; //Distance to 0 so the first one is a min distance
			for(int k = 0; k < n_att; k++) // compute the distance between the two instances
			{
				//mirar para cargar en shared el val j y asi no tener problemas de stride
				diff = attr[idrow* n_att +k] - attr[column*n_att+k];
				distance[idrow * n_inst + column] += diff * diff;
			}
			distance[idrow * n_inst + column] = sqrt(distance[idrow * n_inst + column]);
			if (idrow == column){ // when it is the same point
				distance[idrow * n_inst + column] = FLT_MAX;
			}
			//for(int a = 0; a<n_inst; a++){
			//	for(int b = 0; b<n_inst; b++){
			//		if (row == a && column == b){
			//			printf("element (%d, %d): %f \n",a, b, distance[row * n_inst + column]);
			//		}
			//	}
			//}
	}
}
			//Acabar aqui el kernel, hacer el sort con thrust, y luego otro kernel
__global__ void pred(int *pred, int com, int n_inst, float *distance, float *smallestDistance, int* smallestDistanceClass, int *val, int stream, int elementsstream){
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
	int idx = stream * elementsstream + tid;
	if (tid < elementsstream && idx < n_inst){
		for(int j = 0; j < n_inst; j++){
			for (int n = idx * com ; n<idx * com + com; n++)
				{
					if(distance[n_inst*j+idx] < smallestDistance[n]) // select the closest one
					{
						for (int t=idx * com + com-1; t>n; t--)
						{
							smallestDistance[t] = smallestDistance[t-1];
							smallestDistanceClass[t] = smallestDistanceClass[t-1];
						}
						smallestDistance[n] = distance[n_inst*j+idx];
						smallestDistanceClass[n] = val[j];
						break;
					}
				}
		}
		int freq = 0;
		int predict=0;
		for ( int m = idx * com; m<idx * com + com; m++)
		{
			int tfreq = 1;
			int tpredict=smallestDistanceClass[m];
			for (int s = m+1 ; s< idx * com + com; s++)
			{
				if (tpredict==smallestDistanceClass[s])
				{
					tfreq++;
				}
			}
			if (tfreq>freq)
			{
				predict=smallestDistanceClass[m];
				freq=tfreq;
			}
		}

		pred[idx]= predict;
	}
}

int* KNN(ArffData* dataset, int com)
{
	int n_streams = 4;

	int n_att = dataset->num_attributes() - 1;
	int n_inst = dataset->num_instances();

	int *h_pred= (int*)malloc(n_inst * sizeof(int));
	int *h_val= (int*)malloc(n_inst * sizeof(int));
	float *h_at= (float*)malloc(n_inst * n_att * sizeof(float));


	float *d_at, *d_dist;
	int *d_val, *d_pred;
	cudaMalloc(&d_at, n_inst * n_att * sizeof(float));
	cudaMalloc(&d_dist, n_inst * n_inst * sizeof(float));
	cudaMalloc(&d_val, n_inst* sizeof(int));
	cudaMalloc(&d_pred, n_inst* sizeof(int));

	float* smallestDistance;
	int* smallestDistanceClass;
	cudaMalloc(&smallestDistance,n_inst * com * sizeof(float));
	cudaMalloc(&smallestDistanceClass,n_inst * com * sizeof(int));

	for (int i = 0; i<n_inst; i++){
		h_val[i] = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();

		for( int k = 0; k < n_att; k++){
			h_at[i*n_att+k] = dataset->get_instance(i)->get(k)->operator float();
		}
	}

	//Creating streams
	cudaStream_t *streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));
	for (int i = 0; i < n_streams; i++){
		cudaStreamCreate(&streams[i]);
	}

	int threadperblockdim = 16;
	int elementsperstream = (n_inst + n_streams - 1) / n_streams;
	int griddimx = (n_inst + threadperblockdim - 1) / threadperblockdim;
	int griddimy = (elementsperstream + threadperblockdim - 1) / threadperblockdim;

	dim3 blocksize(threadperblockdim,threadperblockdim);
	dim3 gridsize(griddimx,griddimy);

	int threadperblock = 256;
	int blocks = (elementsperstream + threadperblock - 1) / threadperblock;

	cudaMemcpy(d_val,h_val, n_inst* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_at,h_at, n_att * n_inst* sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < n_streams; i++){
		sKNN<<<gridsize ,blocksize, 0, streams[i]>>>(d_at, d_val, n_att, n_inst, d_dist, com, smallestDistance, i, elementsperstream);
	}

	cudaDeviceSynchronize();

	for (int i = 0; i < n_streams; i++){
		pred<<<blocks ,threadperblock, 0, streams[i]>>>(d_pred, com, n_inst, d_dist, smallestDistance, smallestDistanceClass, d_val, i, elementsperstream);
	}

	for (int i = 0; i < n_streams; i++){
		if (elementsperstream <= (n_inst-i*elementsperstream))
		{
			cudaMemcpyAsync(&h_pred[i*elementsperstream], &d_pred[i*elementsperstream], elementsperstream* sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
		}
		else{
			cudaMemcpyAsync(&h_pred[i*elementsperstream], &d_pred[i*elementsperstream], (n_inst-i*elementsperstream)* sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
			}
	}
	cudaDeviceSynchronize();

	return h_pred;
}


int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
		cout << "Usage: k value" << endl;
        exit(0);
    }

    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;

	int k;

	sscanf(argv[2], "%d", &k);


	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	int* predictions = KNN(dataset,k);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    printf("The KNN classifier  for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
