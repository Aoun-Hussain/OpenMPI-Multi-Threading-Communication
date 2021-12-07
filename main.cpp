/*
Compilations Instructions: Use the following command to run:
                            "qsub -q coc-ice -A YOU_USER_NAME -l nodes=8:ppn=2,walltime=02:00:00 -I"
                            Then load appropriate modules by: module load gcc mvapich2
                            Then run: mpic++ main.cpp -o a.out
                            and finally run:
                            " mpirun -np 16 ./a.out -N 14 -I 10 " OR  " mpirun -np 16 ./a.out -N 254 -I 10000 "

                            PLEASE NOTE THAT VALUE FOR -N should (N+2)%16==0 FOR THE PROGRAM TO WORK
                            All arguments checks are in place.
*/

/*
 References:
    https://dournac.org/info/parallel_heat2d#mpi-implementation
    https://github.com/varunsharma0286/LaplaceEquation/blob/master/OpenMPI%20Solution/mpiRun.c
    https://github.com/gurpal09/Parallel-Computing-2D_Heat_Parallelized
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "unistd.h"
#include <list>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdio>
#include "mpi.h"
#include <cstring>
#include <sstream>

using namespace std;

//value for tage values
#define SEND_UP 1
#define SEND_LOW 1
#define RECV_UP 1
#define RECV_LOW 1

//value for first processor
#define FIRSTP 0

bool isNumeric(const string &strIn, long &nInputNumber)
{
    /*
    Checks if the argument is numeric and returns true/false accordingly
    checks for the arguments -N and -I
    */

    bool bRC = all_of(strIn.begin(), strIn.end(), [](unsigned char c)
    {
        return ::isdigit(c);                      // http://www.cplusplus.com/reference/algorithm/all_of/
    }                                             // https://www.geeksforgeeks.org/lambda-expression-in-c/
    );                                            // http://www.cplusplus.com/reference/cctype/isdigit/

    if (bRC)
    {
        nInputNumber = stoul(strIn);              // https://www.cplusplus.com/reference/string/stoul/
        return true;
    }
    else
    {
        return false;
    }
}

double **alloc2dDouble(int rows, int cols)
{
    /*
    function to contiguously allocate 2D double arrays
    */

    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i = 0; i < rows; i++)
    {
        array[i] = &(data[cols*i]);
    }
    return array;
}

int main(int argc, char* argv[])
{
    /*
    main function to check for all invalid combination of arguments -N and -I
    and performs multi-threaded laplace computations and write the coordinate values to csv file
    and outputs total computation time using MPI
    */

    if (argc == 5)
    {
        long dim{ 0 };
        long iter{ 0 };
        string strInput1(argv[1]);    //-N
        string strInput2(argv[2]);    //positive integer
        string strInput3(argv[3]);    //-I
        string strInput4(argv[4]);    //positive integer

        if ((strInput1 != "-N") || (strInput3 != "-I"))
        {
            cout << "Invalid parameters, please check your values" << endl;
            return EXIT_SUCCESS;
        }

        bool bIsValid1 = isNumeric(strInput2, dim);
        bool bIsValid2 = isNumeric(strInput4, iter);

        if ((bIsValid1) && (bIsValid2) && (!strInput2.empty()) && (!strInput4.empty()) && (dim > 0) && ((dim + 2) % 16 == 0) && (dim <= 256) && (iter >= 1) && (iter <= 10000))
        {
            long X = dim + 2;       //assigning N+2 to X
            long Y = dim + 2;       //assigning N+2 to Y
            long ITER = iter;       //assigning iteration number

            int rank, size;

            MPI_Status status;

            //initializing MPI
            MPI_Init(&argc, &argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            double startTime, endTime;  //variables to store time value for each process
            startTime = MPI_Wtime();

            int Dim = X;                //size of the square grid
            double **matrix;
            double **matrix1;
            double **matrix2;
            double **matrix3;
            long globalSteps = 0;
            int column = Dim;
            int RowPerProcessor;        //store the value of total rows for each processor out of 16
            int last = 0;

            double *upRow;
            double *lowRow;
            long int localSteps = 0;    //variable to store iterations per process
            int startIdx = 0, endIdx;
            int LASTP = size - 1;       //Total number of processes, size will be 16

            if ((Dim % size) == 0)      //THIS HAS TO BE TRUE -> (n+2) % 16 == 0
            {
                RowPerProcessor = Dim/size;
                last = RowPerProcessor;
            }

            if (rank == LASTP)
            {
                RowPerProcessor = last;
            }

            //allocating 2D arrays for all processes
            matrix = alloc2dDouble(RowPerProcessor, column);  //old array
            matrix1 = alloc2dDouble(RowPerProcessor, column);; //new array

            if (rank == 0)
            {
                matrix2 = alloc2dDouble(column, column);  //old array
                matrix3 = alloc2dDouble(column, column);; //new array
            }

            //initializing 2D arrays

            //setting all nodes of plate to 20 C (internal temp)

            if (rank == 0)
            {
                for (int i = 0; i < column; i++)
                {
                    for (int j = 0; j < column; j++)
                    {
                        matrix2[i][j] = 20.0000000000;
                        matrix3[i][j] = 20.0000000000;
                    }
                }
            }

            for (int i = 0; i < RowPerProcessor; i++)
            {
                for(int j = 0; j < column; j++)
                {
                    matrix[i][j] = 20.0000000000;
                    matrix1[i][j] = 20.0000000000;
                }
            }

            //setting 40% of top side to 100 C
            if (rank == FIRSTP)
            {
                for (int j = 0; j < column; j++)
                {
                    if ((j > 0.3 * (column - 1)) && (j < 0.7 * (column - 1)))
                    {
                        matrix[0][j] = 100.0000000000;
                        matrix1[0][j] = 100.0000000000;

                        if (rank == 0)
                        {
                            matrix2[0][j] = 100.0000000000;
                            matrix3[0][j] = 100.0000000000;
                        }
                    }
                }
            }

            //initialization done here for all processes
            MPI_Barrier(MPI_COMM_WORLD);

            upRow = (double *)malloc(sizeof(double) * (column));
            lowRow = (double *)malloc(sizeof(double) * (column));

            //assigns values for start and end indexes appropriately for rows of each processor
            if (rank == LASTP)
            {
                startIdx = 0;
                endIdx = RowPerProcessor - 2;
            }
            else if (rank == FIRSTP)
            {
                startIdx = 1;
                endIdx = RowPerProcessor - 1;
            }
            else
            {
                startIdx = 0;
                endIdx = RowPerProcessor - 1;
            }

            //applies finite difference method on matrix and matrix1 and uses send and receive
            while (true)
            {
                if (rank == FIRSTP)
                {
                    MPI_Send(matrix[RowPerProcessor - 1], column, MPI_DOUBLE, 1, SEND_UP, MPI_COMM_WORLD);
                    MPI_Recv(upRow, column, MPI_DOUBLE, 1, RECV_UP, MPI_COMM_WORLD, &status);
                }
                else if (rank == LASTP)
                {
                    MPI_Send(matrix[0], column, MPI_DOUBLE, size - 2, SEND_LOW, MPI_COMM_WORLD);
                    MPI_Recv(lowRow, column, MPI_DOUBLE, size - 2, RECV_LOW, MPI_COMM_WORLD, &status);
                }
                else
                {
                    MPI_Send(matrix[0], column, MPI_DOUBLE, rank - 1, SEND_LOW, MPI_COMM_WORLD);
                    MPI_Send(matrix[RowPerProcessor - 1], column, MPI_DOUBLE, rank + 1, SEND_UP, MPI_COMM_WORLD);

                    MPI_Recv(lowRow, column, MPI_DOUBLE, rank - 1, RECV_LOW, MPI_COMM_WORLD, &status);
                    MPI_Recv(upRow, column, MPI_DOUBLE, rank + 1, RECV_UP, MPI_COMM_WORLD, &status);
                }
                MPI_Barrier(MPI_COMM_WORLD);

                startIdx = endIdx;

                for (int i = startIdx; i < endIdx; i++)
                {
                    for (int j = 1; j < column; j++)
                    {
                        if ((rank == FIRSTP) && (i == endIdx)) //First process
                        {
                            matrix1[i][j] = (matrix[i-1][j] + upRow[j] + matrix[i][j-1] + matrix[i][j+1])/4.0;
                        }
                        else if ((rank == LASTP) && (i == startIdx)) //Last process
                        {
                            matrix1[i][j] = (matrix[i+1][j] + lowRow[j] + matrix[i][j-1] + matrix[i][j+1])/4.0;
                        }
                        else if (((i == startIdx) || (i == endIdx)) && (rank > FIRSTP) && (rank < LASTP))
                        {
                            if(i == startIdx)
                            {
                                matrix1[i][j] = (lowRow[j] + matrix[i+1][j] + matrix[i][j-1] + matrix[i][j+1])/4.0;
                            }
                            else if (i == endIdx)
                            {
                                matrix1[i][j] = (matrix[i-1][j] + upRow[j] + matrix[i][j-1] + matrix[i][j+1])/4.0;
                            }
                        }
                        else
                            matrix1[i][j] = (matrix[i-1][j] + matrix[i+1][j] + matrix[i][j-1] + matrix[i][j+1])/4.0;
                    }
                }

                if (rank == 0)
                {
                    while (globalSteps != ITER)
                    {
                        for (int i = 1; i < column - 1; i++)
                        {
                            for (int j = 1; j < column - 1; j++)
                            {
                                matrix3[i][j] = (matrix2[i-1][j] + matrix2[i+1][j] + matrix2[i][j-1] + matrix2[i][j+1])/4.0;
                            }
                        }

                        for (int i = 1; i < column - 1; i++)
                        {
                            memcpy(matrix2[i], matrix3[i],sizeof(double) * column);
                        }
                        globalSteps ++;
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);

                //swaps the row, updates the old with new, keep ready for next iteration
                for (int i = 0; i < RowPerProcessor; i++)
                {
                    memcpy(matrix[i], matrix1[i],sizeof(double) * column);   //copy matrix1 to matrix after each iteration
                }
                MPI_Barrier(MPI_COMM_WORLD);

                localSteps++;           //increase steps after each process iteration

                //checks if total iterations have been met
                if (localSteps == ITER)
                {
                    endTime = MPI_Wtime();
                    break;                 //all processes break after number of iterations are met
                }
            }

            if (rank == 0)                                         //returning total time consumed by master*size
            {
                double tme = (endTime - startTime);   //storing time consumed
                double mul = pow(10.0, 2);                //rounding off to 2 decimal places
                float tm = ceil(tme * mul) / mul;
                cout << tm << endl;
            }

            //each process writes its array values inside the file
            //FOR DEBUGGING PURPOSE
//            string f_name;
//            f_name = "csv_" + to_string(rank);
//            ofstream myfile(f_name);
//            if (myfile.is_open())
//            {
//                for (int k = 0; k < RowPerProcessor; k++)
//                {
//                    string line;
//                    for (int l = 0; l < column; l++)
//                    {
//                        //rounding off to 10 decimal places
//                        double multiplier = pow(10.0, 10);
//                        double val = ceil(matrix1[k][l] * multiplier) / multiplier;
//                        string elem = to_string(val) + ",";
//                        line.append(elem);
//                    }
//                    myfile << line << "\n"; //comma separated values and new line after every row
//                }
//                myfile.close();
//            }
//            else
//            {
//                cout << "Unable to open file";
//                return EXIT_SUCCESS;
//            }

            //free pointers
//            free(matrix[0]);
//            free(matrix);
//            free(matrix1[0]);
//            free(matrix1);

            double **finMatrix = nullptr;                 //declares the final matrix

            if (rank == 0)
            {
                finMatrix = alloc2dDouble(column, column); //allocating final 2D array
            }

            //gathering all values in the root process
            MPI_Gather(&matrix1[0], 1, MPI_DOUBLE, finMatrix, 1, MPI_DOUBLE, FIRSTP, MPI_COMM_WORLD);


            //using root process to print the final csv file
            if (rank == 0)
            {
                //writing to final csv file
                ofstream myfile("finalTemperatures.csv");
                if (myfile.is_open())
                {
                    for (int k = 0; k < column; k++)
                    {
                        string line;
                        for (int l = 0; l < column; l++)
                        {
                            //rounding off to 10 decimal places
                            double multiplier = pow(10.0, 10);
                            double val = ceil(matrix3[k][l] * multiplier) / multiplier;
                            //double val = ceil(finMatrix[k][l] * multiplier) / multiplier;
                            string elem = to_string(val) + ",";
                            line.append(elem);
                        }
                        myfile << line << "\n"; //comma separated values and new line after every row
                    }
                    myfile.close();
                }
                else
                {
                    cout << "Unable to open file";
                    return EXIT_SUCCESS;
                }
            }

//            if (rank == 0)
//            {
//                //free pointers
//                free(finMatrix[0]);
//                free(finMatrix);
//            }

            //end MPI
            MPI_Finalize();
            return 0;
        }
        else
        {
            cout << "Invalid parameters, please check your values" << endl;
            return EXIT_SUCCESS;
        }
    }
    else
    {
        cout << "Invalid parameters, please check your values" << endl;
        return EXIT_SUCCESS;
    }
}









