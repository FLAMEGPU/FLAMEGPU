/*
This programs geneates the .xml input file.
Author : Mozhgan K. Chimeh, Paul Richmond

To Compile: g++ inpGen.cpp -D CASE1 -o inpGen
To Execute: ./inpGen iterations/1.xml 10
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define L_MAX 2.0
#define NL_MAX 0.4
#define BIN_WIDTH 0.1
#define BIN_COUNT L_MAX/BIN_WIDTH 

int hist[BIN_COUNT];

int main( int argc, char** argv) 
{
	srand48(time(NULL));

	int n_c = atoi(argv[2]);
	char * fileName = argv[1];
	int n = 0;

	for (int i=0; i<BIN_COUNT; i++){
		hist[i] = 0;
	}

	FILE *fp = fopen(fileName, "w"); // write only 
	   
	// test for files not existing. 
	if (fp== NULL) {   
		printf("Error! Could not open file\n"); 
		exit(-1); // must include stdlib.h 
	} 

	fprintf(fp, "<state>\n");
	fprintf(fp, "<itno>0</itno>\n");

	while (n < n_c){
		double l_temp = drand48()*L_MAX;
		double nl_temp = drand48()*NL_MAX; 
		if (n_c*pow(l_temp, 2)*exp(-1*pow(l_temp, 3)) > nl_temp){
			fprintf(fp, "<xagent>\n<name>crystal</name>\n<rank>0</rank>\n<l>%f</l>\n</xagent>\n",l_temp);
			int bin = l_temp / BIN_WIDTH;
			if (bin >= BIN_COUNT){
				printf("Somthing bad happened!\n");
			}
			hist[bin]++;
			n++;
		}

	}

	fprintf(fp, "</state>");
	fclose(fp);

	//output the hist in some format useful to gnuplot or whatever

	return 0;
}
