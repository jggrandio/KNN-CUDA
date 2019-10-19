#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
using namespace std;

__global__ void sKNN(float *attr, int *val, int n_att, int n_inst, float *distance, int com, float *smallestDistance){

	//calculate tid
	extern __shared__ float sh[];
	int column = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
	int tid = (blockDim.x*gridDim.x*row)+column;

	if (tid < n_inst * com){
		smallestDistance[tid] = FLT_MAX;
	}

		if (column < n_inst * n_att && row < n_inst){
			if(threadIdx.x == 0){
				distance[row * n_inst + blockIdx.x] = 0; //Distance to 0 so the first one is a min distance
			}

			float diff;

			diff = (attr[row* n_att + threadIdx.x] - attr[column])*(attr[row* n_att + threadIdx.x] - attr[column]);
			sh[threadIdx.y*n_att+threadIdx.x]=diff;
			__syncthreads();

			if(threadIdx.x == 0){

				for (int k = threadIdx.y*n_att; k < (threadIdx.y*n_att) + n_att; k++){


					distance[row * n_inst + blockIdx.x] += sh[k];

				}

				distance[row * n_inst + blockIdx.x] =  sqrt(distance[row * n_inst + blockIdx.x]); //Distance to 0 so the first one is a min distance


				if (row == blockIdx.x){ // when it is the same point
					distance[row * n_inst + blockIdx.x] = FLT_MAX;
				}
			}

			//for(int a = 0; a<n_inst; a++){
			//	for(int b = 0; b<n_inst; b++){
			//		if (row == a && column == b){
			//			printf("element (%d, %d): %f \n",a, b, distance[row * n_inst + column/n_att]);
			//		}
			//	}
			//}
		}
}

__global__ void pred(int *pred, int com, int n_inst, float *distance, float *smallestDistance, int* smallestDistanceClass, int *val){
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (tid < n_inst){
		for(int j = 0; j < n_inst; j++){
			for (int n = tid * com ; n<tid * com + com; n++)
				{
					if(distance[n_inst*j+tid] < smallestDistance[n]) // select the closest one
					{
						for (int t=tid * com + com-1; t>n; t--)
						{
							smallestDistance[t] = smallestDistance[t-1];
							smallestDistanceClass[t] = smallestDistanceClass[t-1];
						}
						smallestDistance[n] = distance[n_inst*j+tid];
						smallestDistanceClass[n] = val[j];
						break;
					}
				}
		}
		int freq = 0;
		int predict=0;
		for ( int m = tid * com; m<tid * com + com; m++)
		{
			int tfreq = 1;
			int tpredict=smallestDistanceClass[m];
			for (int s = m+1 ; s< tid * com + com; s++)
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

		pred[tid]= predict;
	}
}

int* KNN(ArffData* dataset, int com)
{
	int n_att = dataset->num_attributes() - 1;
	int n_inst = dataset->num_instances();

	int threadperblockdim = n_att;
	int griddimx = ((n_att*n_inst) + threadperblockdim - 1) / threadperblockdim;
	int griddimy = (n_inst + threadperblockdim - 1) / threadperblockdim;

	dim3 blocksize(threadperblockdim,threadperblockdim);
	dim3 gridsize(griddimx,griddimy);


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
	cudaMemcpy(d_val,h_val, n_inst* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_at,h_at, n_att * n_inst* sizeof(float), cudaMemcpyHostToDevice);
	sKNN<<<gridsize , blocksize,n_att*n_att*sizeof(float)>>>(d_at, d_val, n_att, n_inst, d_dist, com, smallestDistance);


	int threadperblock = 256;
	int blocks = (dataset->num_instances() + threadperblock - 1) / threadperblock;
	pred<<<blocks , threadperblock>>>(d_pred, com, n_inst, d_dist, smallestDistance, smallestDistanceClass, d_val);

	cudaMemcpy(h_pred, d_pred, n_inst* sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_at);
	cudaFree(d_dist);
	cudaFree(d_val);
	cudaFree(d_pred);
	cudaFree(smallestDistance);
	cudaFree(smallestDistanceClass);

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
