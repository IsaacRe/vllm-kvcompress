rm -f ./a.out
nvcc $1
./a.out $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}
compute-sanitizer --tool memcheck ./a.out $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}