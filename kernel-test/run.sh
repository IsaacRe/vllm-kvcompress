rm -f ./a.out
nvcc $1
./a.out $2 $3 $4 $5 $6 $7
compute-sanitizer --tool memcheck ./a.out $2 $3 $4 $5 $6 $7