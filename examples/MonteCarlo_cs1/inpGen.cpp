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

static double CND(double d)
{

    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

 return cnd;
}

double normalCFD(double value){}

int main( int argc, char** argv) 
{
srand48(time(NULL));

int iter = atoi(argv[2]);
char * fileName = argv[1];
      
FILE *fp = fopen(fileName, "w"); // write only 
           
// test for files not existing. 
if (fp== NULL) {   
  printf("Error! Could not open file\n"); 
  exit(-1); // must include stdlib.h 
} 

fprintf(fp, "%s \n", "<state>");
fprintf(fp, "%s \n", "<itno>0</itno>");

for (int i=0; i<iter ; i++){

  double l= CND(drand48());

#ifdef CASE2
   fprintf(fp, "<xagent><name>crystal</name><l>%f</l></xagent>", l);
#endif

#ifdef CASE1
   fprintf(fp, "<xagent><name>crystal</name><rank>%d</rank><l>%f</l></xagent>",iter+1,l);
#endif

}
fprintf(fp, "%s \n", "</state>");
fclose(fp);

return 0;
}
