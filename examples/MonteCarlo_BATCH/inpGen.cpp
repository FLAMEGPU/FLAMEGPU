/*
This programs geneates the .xml input file.
Author : Mozhgan K. Chimeh, Paul Richmond

To Compile: g++ inpGen.cpp -o inpGen
To Execute: ./inpGen iterations/0.xml ~/Desktop/input.dat 10000
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define L_MAX 2.0
#define NL_MAX 0.4
#define BIN_WIDTH 0.1
#define BIN_COUNT L_MAX/BIN_WIDTH

int hist[int(BIN_COUNT)];

int main( int argc, char** argv)
{
    srand48(time(NULL));

    int n_c = atoi(argv[3]);
    char * fileName1 = argv[1];
    char * fileName2 = argv[2];
    int n = 0;

    for (int i=0; i<BIN_COUNT; i++) {
        hist[i] = 0;
    }

    FILE *fp = fopen(fileName1, "w"); // write only
    FILE *hist_output = fopen(fileName2, "w"); // write only

    // test for files not existing.
    if (fp== NULL) {
        printf("Error! Could not open file\n");
        exit(-1); // must include stdlib.h
    }

    fprintf(fp, "<states>\n");
    fprintf(fp, "<itno>0</itno>\n");

    while (n < n_c) {
        double l_temp = drand48()*L_MAX;
        double nl_temp = drand48()*NL_MAX;
        if (pow(l_temp, 2)*exp(-1*pow(l_temp, 3)) > nl_temp) { //n_c* -- we removed the n_c
            fprintf(fp, "<xagent><name>crystal</name><rank>0</rank><l>%f</l></xagent>\n",l_temp);
            int bin = l_temp / BIN_WIDTH;
            if (bin >= BIN_COUNT) {
                printf("Error, bin > BIN_COUNT!!!\n");
            }
            hist[bin]++;
            n++;
        }

    }

    fprintf(fp, "</states>");
    fclose(fp);

    for (int i=0; i<BIN_COUNT; i++) {
//output the hist in to *.dat, ready to plot
        fprintf(hist_output,"%f %d\n", i*BIN_WIDTH, hist[i]);
    }

    fclose(hist_output);


    return 0;
}
