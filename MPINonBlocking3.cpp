// Demonstrate simple MPI program
// This one uses non-blocking receives and the tag field.
// George F. Riley, Georgia Tech, Fall 2011


#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

using namespace std;

int main(int argc, char**argv)
{
    int  numtasks, rank, rc;

    rc = MPI_Init(&argc, &argv);

    if (rc != MPI_SUCCESS) 
    {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Number of tasks= %d My rank= %d\n", numtasks, rank);
    int w = 256;
    int h = 256;
    int sz = w * h;
    int each = sz / numtasks;
    int myStart = rank * each;
    // Create the array and populate each index value with the index
    int* pArray = new int[sz];
    for (int i = 0; i < sz; ++i)
    {
        pArray[i] = i;

    }
    // The target array is uninitialized, filled in by the
    // successful MPI_Irecv callls.
    int* pTarget = new int[sz];
    // We also need an array of MPI_Request variables, one for each
    // pending Irecv.
    MPI_Request* pReq = new MPI_Request[sz];

    // If rank 0, queue "sz" non-blocking receivs with the tag being
    // the array index and the target buffer being the address
    // of pTarget[i].
    if (rank == 0)
    {
        for (int i = 0; i < sz; ++i)
        {
            rc = MPI_Irecv(&pTarget[i], 1, MPI_INT, MPI_ANY_SOURCE,
                i, MPI_COMM_WORLD, &pReq[i]);
        }
    }
    // Send results to cpu 0, (note cpu0 sends to itself)
    int k = each;
    while (k > 0)
    {
        k--;
        MPI_Send(&pArray[myStart + k], 1, MPI_INT, 0, myStart + k, MPI_COMM_WORLD);
    }
    // for (int i = 0; i < each; ++i)
    //   {
    //     MPI_Send(&pArray[myStart + i], 1, MPI_INT, 0, myStart + i, MPI_COMM_WORLD);
    //   }
    if (rank == 0)
    {
        for (int i = 0; i < sz; ++i)
        {
            MPI_Status status;
            MPI_Wait(&pReq[i], &status);
            cout << "CPU 0 got index " << i
                << " value " << pTarget[i] << endl;
        }
    }
    cout << "Rank " << rank << " exiting normally" << endl;
    MPI_Finalize();
}

