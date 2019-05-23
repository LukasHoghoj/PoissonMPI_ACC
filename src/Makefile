#----------------------------------------------------------------------
#                                                                      
#  Makefile : diffuse2     
#  Version  :                                                          
#  Author   :                                                          
#  Created  : 
#                                                                      
#----------------------------------------------------------------------
#  Creator: makemake 2.1
#  Rev: 1996/01/11 by Jens H. Walther  
#----------------------------------------------------------------------
SHELL       = /bin/sh
TARGET      = jacobi
#----------------------------------------------------------------------
#  Compiler settings (Linux)
#----------------------------------------------------------------------
F77         = mpif90
CC          = mpicc
CUDA        = cuda9.0
DEBUG       = 
DEBUG       = 
OPT         = 
FFLAGS      = $(OPT) $(DEBUG)



CFLAGS      = -fast
LD          = $(CC)
LDFLAGS     =  
CPP         = /lib/cpp
DEFINE      = 
LIBS        = -lm
ACC         = -acc -ta=tesla:$(CUDA) -Mlarge_arrays

CFLAGS+= $(ACC) -Minfo=accel,mp 
LDFLAGS+= $(ACC)


#----------------------------------------------------------------------
#  Search path for RCS files                                           
#----------------------------------------------------------------------
VPATH = ./RCS

#----------------------------------------------------------------------
#  Additional suffix rules                                             
#----------------------------------------------------------------------
.SUFFIXES : .inc .inc,v .f,v .c,v
.f,v.f :
	 co $*.f

.c,v.c :
	 co $*.c

.inc,v.inc :
	 co $*.inc

#----------------------------------------------------------------------
#  Binary directory
#----------------------------------------------------------------------
bindir      = $(HOME)/bin

#----------------------------------------------------------------------
#  Default target
#----------------------------------------------------------------------
all: $(TARGET)

#----------------------------------------------------------------------
#  Object files:                                                       
#  NOTE: you HAVE to sort the objects files such that no file will 
#  depend on files below it ! in this example, the diffuse2.f and .o
#  depends on all he module files (i named them m_*.f), and the m_init
#  depends (USE) the m_diffuse; thus m_diffuse HAS to be compiled 
#  before m_init and before diffuse2
#----------------------------------------------------------------------
OBJS =\
	initACC.o\
	ST.o\
	setBC_IC.o\
	poisson3D_serial.o\
	poisson3D_ACC1.o\
	poisson3D_ACC2.o\
	poisson3D_MPI1.o\
	poisson3D_MPI2.o\
	poisson3D_hyb1.o\
	poisson3D_hyb2.o\
	poisson3D_hyb3.o\
	poisson3D_hyb4.o\
	poisson3D_hyb5.o\
	mat2file.o\
	printres.o\
	jacobi.o

#-------------------------------------------------------------------
#  Dependencies:                                                       
#  NOTE: add the dependencies here explicitly ! 
#  In that way you are sure diffuse2.f will be recompile if any of the
#  modules source files are modified.
#----------------------------------------------------------------------
initACC.o: initACC.c
	$(CC) $(CFLAGS) -c initACC.c

ST.o: ST.c
	$(CC) $(CFLAGS) -c ST.c

setBC_IC.o: setBC_IC.c
	$(CC) $(CFLAGS) -c setBC_IC.c

poisson3D_serial.o: poisson3D_serial.c
	$(CC) $(CFLAGS) -c poisson3D_serial.c

poisson3D_ACC1.o: poisson3D_ACC1.c
	$(CC) $(CFLAGS) -c poisson3D_ACC1.c

poisson3D_ACC2.o: poisson3D_ACC2.c
	$(CC) $(CFLAGS) -c poisson3D_ACC2.c

poisson3D_MPI1.o: poisson3D_MPI1.c
	$(CC) $(CFLAGS) -c poisson3D_MPI1.c

poisson3D_MPI2.o: poisson3D_MPI2.c
	$(CC) $(CFLAGS) -c poisson3D_MPI2.c

poisson3D_hyb1.o: poisson3D_hyb1.c
	$(CC) $(CFLAGS) -c poisson3D_hyb1.c

poisson3D_hyb2.o: poisson3D_hyb2.c
	$(CC) $(CFLAGS) -c poisson3D_hyb2.c

poisson3D_hyb3.o: poisson3D_hyb3.c
	$(CC) $(CFLAGS) -c poisson3D_hyb3.c

poisson3D_hyb4.o: poisson3D_hyb4.c
	$(CC) $(CFLAGS) -c poisson3D_hyb4.c

poisson3D_hyb5.o: poisson3D_hyb5.c
	$(CC) $(CFLAGS) -c poisson3D_hyb5.c

mat2file.o: mat2file.c
	$(CC) $(CFLAGS) -c mat2file.c

printres.o: printres.c
	$(CC) $(CFLAGS) -c printres.c

jacobi.o: jacobi.c initACC.c ST.c setBC_IC.c poisson3D_serial.c poisson3D_ACC1.c poisson3D_ACC2.c poisson3D_hyb1.c poisson3D_hyb2.c poisson3D_hyb3.c poisson3D_hyb4.c poisson3D_hyb5.c poisson3D_MPI1.c mat2file.c
	$(CC) $(CFLAGS)  -c jacobi.c
#----------------------------------------------------------------------
#  link                                                                
#----------------------------------------------------------------------
$(TARGET): $(OBJS)
	$(LD) -o $(TARGET) $(LDFLAGS) $(OBJS) $(LIBS)

#----------------------------------------------------------------------
#  Install                                                             
#----------------------------------------------------------------------
install: $(TARGET)
	(cp -f $(TARGET) $(bindir))

#----------------------------------------------------------------------
#  Run                                                                 
#----------------------------------------------------------------------
run: $(TARGET)
	$(TARGET)

#----------------------------------------------------------------------
#  Clean                                                               
#----------------------------------------------------------------------
new: cleanall $(TARGET)
cleanall:
	 rm -f __*.f
	 rm -f $(OBJS)
	 rm -f *.lst
	 rm -f *.mod
	 rm -f *.l
	 rm -f *.L
	 rm -f *.dat
	 rm -f *.log
	 rm -f $(TARGET)

clean:
	 rm -f __*.f
	 rm -f *.lst
	 rm -f $(OBJS)
	 rm -f *.mod
	 rm -f $(TARGET)
