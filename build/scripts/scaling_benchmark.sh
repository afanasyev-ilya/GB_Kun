args=("$@")
number_of_arguments=$#

threads_count=${args[0]}
program_name=${args[1]}

for (( c=2; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

export OMP_NUM_THREADS=${threads_count}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

${program_name} ${program_args[@]}