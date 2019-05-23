# PoissonMPI_ACC
## Lukas Høghøj - May 2018. 02616 Large Scale Modelling - Technical University of Denmark  
Solving the 3-Dimensional Poisson equation using multiple GPUs. GPU acceleration is enabled by OpenACC and communication by MPI.  

Compiled using the PGI compiler linked with OpenACC and OpenMPI.

### Folders: ###
- src:  
	- 	Contains source code, header files and Makefile. Functions for solving the poisson equation are 
		poisson3D_*.c  and main script is jacobi.c.
- submit:
	-	Contains batch scripts to submit jobs to the DTU GBar and a wrapper script, to ensure correct binding
		between MPI rank and GPUs when running with more than one GPU/node.
- util:
	-	Contains postprocessing python script and timing data from benchmark runs.


Target is ./jacobi  


### Command  line arguments:  ###
- Mandatory input:  
	- 	Experiment type  
		- 	0:  Serial version  
		- 	1:  OpenACC version 1 - very simple    
		- 	2:  OpenACC version 2 - Reduced data movement    
		- 	10: MPI version 1 - Blocking communication    
		- 	11: MPI version 2 - Non - blocking communication   
		- 	20: MPI + OpenACC version 1 - very simple    
		- 	21: MPI + OpenACC version 2 - First data movement reduction, copy one array back and forth to CPU at each iter   
		- 	22: MPI + OpenACC version 3 - Second data movement reduction, only copy relevant data    
		- 	23: MPI + OpenACC version 4 - Non-blocking communication    
		- 	24: MPI + OpenACC version 5 --- NOT WORKING CORRECTLY - Communicate directly from GPU ???   
	- 	Nx 
	- 	Ny 
	- 	Nz   
	- 	Number of iterations  
- Optional input:  
	- 	-o "solution output file"  
	- 	-p "timing output file (append)"  
  
Consider using the submit/wrapper to ensure a correct binding between MPI ranks and GPUs.
