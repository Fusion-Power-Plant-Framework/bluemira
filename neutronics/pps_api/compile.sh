$CXX -Wall -c -g -fPIC prn_c.cpp -I/openmc/include/  -I/miniconda3/include/
$FC -ffree-line-length-none -fPIC -c plasma_lib.F90 plasma_lib_api.F90
ar cr libppsmods.a plasma_lib.o plasma_lib_api.o prn_c.o
$CXX -Wall -c -g -fPIC plasma_OpenMC.cpp -I/openmc/include/ -I/miniconda3/include/
$CXX -o PPS_OpenMC.so plasma_OpenMC.o -L./ -lppsmods -shared -lgfortran
