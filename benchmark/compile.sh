args=("$@")
number_of_arguments=$#

program_name=${args[0]}

cd ../build
make ${program_name}
cd -