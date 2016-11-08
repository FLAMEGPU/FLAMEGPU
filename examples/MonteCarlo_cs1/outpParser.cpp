/*
This programs geneates the .xml input file.
Author : Mozhgan K. Chimeh, Paul Richmond

To Compile: g++ outpParser.cpp -o outpParser
To Execute: ./outpParser iterations/1.xml 1_.xml
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 1024

int main( int argc, char** argv) 
{

char * inpName = argv[1];
char * outpName = argv[2];

const char *delimiter_characters = " < >"; // < >

FILE *input_file = fopen( inpName, "r" );

char buffer[ BUFFER_SIZE ];
char *last_token;

FILE *output_file = fopen(outpName, "w"); // write only 
           
// test for files not existing. 
if (input_file == NULL || output_file == NULL) {   
  fprintf( stderr, "Unable to open file %s\n", inpName );
} 

else{

        // Read each line into the buffer
        while( fgets(buffer, BUFFER_SIZE, input_file) != NULL ){

            // Write the line to stdout
            //fputs( buffer, stdout );

            // Gets each token as a string and prints it
            last_token = strtok( buffer, delimiter_characters );
            while( last_token != NULL ){
                bool p = false;
                printf( "%s\n", last_token );
                if (strcmp(last_token,"l")==0){

                    p=true;
                }
                last_token = strtok( NULL, delimiter_characters );
                if (p){
                    fprintf (output_file,"%s ",last_token);
                    p=false;
                }
            }
        }

        if( ferror(input_file) ){
            perror( "The following error occurred" );
        }

        fclose( input_file );
        fclose( output_file );
    }

    return 0;
}
