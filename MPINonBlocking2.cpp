// Demonstrate simple MPI barriers and collectives
// This one uses non-blocking ISend/Irecv
// George F. Riley, Georgia Tech, Fall 2011


#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

using namespace std;

int main(int argc, char**argv)
{
    int  numtasks, rank, rc;

    // As always, we must call MPI_Init
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Get information about the number of tasks and which
    // rank this task is.
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Seed the RNG with our rank
    srand48(rank * 100 + 10);  // Any value will do as long as different per cpu

    printf("Number of tasks= %d My rank= %d\n", numtasks, rank);

    for (int round = 0; round < 3; ++round)
    { // Each task delays for an amount of time and then barrier's
        double delaySecs = drand48() * 10.0;
        int    sleepSecs = (int)delaySecs;
        cout << "Rank " << rank
            << " delaying for " << sleepSecs << " seconds" << endl;

        sleep(sleepSecs);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    // Now each rank chooses a random value and distributed to all
    // other ranks using allGather
    int    groupSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &groupSize);
    double* pGatherBuffer = new double[groupSize];
    // Set my own value in the buffer
    double myValue = drand48();
    cout << "Rank " << rank << " reporting value " << myValue
        << " groupSize " << groupSize
        << endl;

    MPI_Allgather(&myValue, 1, MPI_DOUBLE, // These 3 are my data
        pGatherBuffer, 1, MPI_DOUBLE, // Receive buffer
        MPI_COMM_WORLD);

    // To reduce amount of output, only rank 0 reports the results
    if (rank == 0)
    {
        for (int i = 0; i < numtasks; ++i)
        {
            cout << "Rank " << i << " reports " << pGatherBuffer[i]
                << endl;
        }
    }
    // Finally try MPI_Allreduce to get a global minimum
    double minValue = 0;  // Global min calculated by allreduce
    MPI_Allreduce(&myValue, &minValue, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    cout << "Rank " << rank << " exiting normally, global min is "
        << minValue << endl;
    MPI_Finalize();
}
