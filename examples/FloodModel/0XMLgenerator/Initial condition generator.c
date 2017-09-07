#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// This function generate 0.XML file based on the data of three hump sulution (FV)
// The output is the initial file to be employed in the flood modelling in FLAME-GPU
// The data which is used in this function is exactly the same as MATLAB solution for SWEs (FV)

float bed_data(float x_int , float y_int);

float initial_flow(float x_int , float y_int , float z0_int);

float maxi(float num1, float num2);

int main()
{
    FILE *fp = fopen("output.txt", "w");
    if (fp == NULL)
    {
    printf("Error opening file!\n");
    exit(1);
    }
    
         // Model constant :  they are imported to the output file          

         float       TIMESTEP = 0.5; // assigned Temporary 
         float       DXL;
         float       DYL;
         int         inDomain = 1 ;
           
        int xmin = 0;
        int xmax = 75;
        int ymin = 0;
        int ymax = 30;
        
        int nx = 46;
        int ny = 16;
        
        float Lx , Ly;
        float dx , dy;
        int i , j ;

          /*Initial variables*/         
          float qx_int[nx+1][ny+1], qy_int[nx+1][ny+1], h_int[nx+1][ny+1], z0_int[nx+1][ny+1];
          float x_int[nx+1],y_int[ny+1];
        
          float qx[nx+2][ny+2], qy[nx+2][ny+2], h[nx+2][ny+2], z0[nx+2][ny+2];
    
          float x[nx+2],y[ny+2];
          float xi[nx],yj[ny]; //In cse a plot is needed

    
    // initial flow rate
    float qx_initial = 0.00;
    float qy_initial = 0.00;
    
    // Mesh-grid propertise
    Lx = xmax - xmin;
    Ly = ymax - ymin;
    dx = (float)Lx/(float)nx;
    dy = (float)Ly/(float)ny;
    
    DXL = (float)dx; // Temporary
    DYL = (float)dy; // Temporary
    
    
    fprintf(fp,"<states>\n");
    fprintf(fp,"<itno>0</itno>\n");
    fprintf(fp," <environment>\n"); 
    fprintf(fp,"  <TIMESTEP>%f</TIMESTEP>\n",TIMESTEP);
    fprintf(fp,"  <DXL>%f</DXL>\n",DXL);
    fprintf(fp,"  <DYL>%f</DYL>\n",DYL);
    fprintf(fp," </environment>\n");
    

      for ( i=1 ; i < nx+1 ; i++){  
          for ( j=1 ; j < ny+1 ; j++){
            
            x_int[i] = xmin + (i-1) * dx; 
            y_int[j] = ymin + (j-1) * dy;
            
            z0_int[i][j] = bed_data    ((float)x_int[i],(float)y_int[j]);
            h_int[i][j]  = initial_flow((float)x_int[i],(float)y_int[j],(float)z0_int[i][j]);
            qx_int[i][j] = qx_initial; // Temporary assigned value
            qy_int[i][j] = qy_initial; // Temporary assigned value (However it should be 0 )
            
//            printf("The value of h_initial in x_interface[%f] y_interface[%f] %3f\n",x_int[i],y_int[j], h_int[i][j] );
            //printf("The value of z0_initial in x_interface[%f] y_interface[%f] %3f\n", x_int[1], y_int[1] , z0_int[1][1] );
                     }
            }
            
            for ( i=2 ; i < nx+1 ; i++){  
                for ( j=2 ; j < ny+1 ; j++){
                    
                    x[i] = 0.5*(x_int[i] + x_int[i-1]);
                    y[j] = 0.5*(y_int[j] + y_int[j-1]);
                    
                    z0[i][j] = 0.25*(z0_int[i][j-1] + z0_int[i][j] + z0_int[i-1][j] + z0_int[i-1][j-1]);
                    h[i][j] = 0.25*(h_int[i][j-1] + h_int[i][j] + h_int[i-1][j] + h_int[i-1][j-1]);
                    
                    qx[i][j] = 0.25*(qx_int[i][j-1] + qx_int[i][j] + qx_int[i-1][j] + qx_int[i-1][j-1]);
                    qy[i][j] = 0.25*(qy_int[i][j-1] + qy_int[i][j] + qy_int[i-1][j] + qy_int[i-1][j-1]);
                   
//                   printf("The value of z0 in x[%f] y[%f] %3f\n", x[i], y[j] , z0[i][j] );
//                   *To test the results : 
                    // fprintf(fp," x[%d] = %.3f\ty[%d] = %.3f\tz0 = %.3f \n ", i, x[i], j , y[j] , z0[i][j] );
                      
                    fprintf(fp, " <xagent>\n");
	                fprintf(fp, "\t<name>FloodCell</name>\n");
	                
	                fprintf(fp, "\t<inDomain>%d</inDomain>\n", inDomain);
                    fprintf(fp, "\t<x>%.4f</x>\n", x[i]);
                    fprintf(fp, "\t<y>%.4f</y>\n", y[j]);
                    fprintf(fp, "\t<z0>%.4f</z0>\n",z0[i][j]);
                    fprintf(fp, "\t<h>%.4f</h>\n",h[i][j]);
                    fprintf(fp, "\t<qx>%.4f</qx>\n",qx[i][j]);
                    fprintf(fp, "\t<qy>%.4f</qy>\n",qy[i][j]);         
                    fprintf(fp, " </xagent>\n");
                   
                    
                                    }
                }
                
                  fprintf(fp, "</states>");          
                  fclose(fp);
                  return 0;
    
}

/* Function to generate the terrain detail - Three Humps*/
 float bed_data(float x_int , float y_int)
 {
       float zz;
       
       float x1 = 30.000;
       float y1 = 6.000;
       float x2 = 30.000;
       float y2 = 24.000;
       float x3 = 47.500;
       float y3 = 15.000;
       
       float rm1 = 8.000;
       float rm2 = 8.000;
       float rm3 = 10.000; 
       
       float r1,r2,r3;
       float zb1,zb2,zb3,zb4;
       float zz1,zz2;
       
       float x01,x02,x03,y01,y02,y03;


       x01 = x_int - x1;
       x02 = x_int - x2;
       x03 = x_int - x3;
       y01 = y_int - y1;
       y02 = y_int - y2;
       y03 = y_int - y3;
       
//       r1 = pow(((pow(x01,2))+(pow(y01,2))),0.5);
//       r2 = pow(((pow(x02,2))+(pow(y02,2))),0.5);
//       r2 = pow(((pow(x03,2))+(pow(y03,2))),0.5);
       
       r1 = sqrt((pow(x01,2.0)) + (pow(y01,2.0)));
       r2 = sqrt((pow(x02,2.0)) + (pow(y02,2.0)));
       r3 = sqrt((pow(x03,2.0)) + (pow(y03,2.0))); 

       
       zb1 = (rm1 - r1)/8 ;
       zb2 = (rm2 - r2)/8 ;
       zb3 = 3 * (rm1 - r3)/10 ;
       zb4 = 0; /*This is the max surface*/
       
       zz1 = maxi((float)zb1,(float)zb2 );
       zz2 = maxi((float)zb3,(float)zb4 );
       
       zz = 1.0 * maxi((float)zz1 , (float)zz2);
       
       return zz;
       }
       
 /* function returning the max between two numbers */
float maxi(float num1, float num2) {
      
        float result;
    
      if (num1 > num2)
       result = num1;
       else
           result = num2;
 
   return result; 
}  
 
 float initial_flow(float x_int , float y_int , float z0_int)
 {

       float etta = 1.875;
       float h;
       
       if ( x_int <= 16 ) {
            h = etta - z0_int;
            }
            else{
                 h = 0;
                 }
                 
          return h;
          }
                  
       
