for i in 0 1 2 3 4 5 6 7 8 9
do
    name=WZ_HyperOpt_NN_${i}
    echo "bash launch_NN.sh |& tee $PWD/logs/$name.LOG" \
	| qsub -v "NAME=$name,DATA_PREFIX=$data_prefix,TYPE=$type" \
	-N $name \
	-d $PWD \
	-l nice=0 \
	-j oe \
	-o $PWD/logs \
	-e $PWD/logs \
	-l nodes=1:ppn=2 \
	-q atlas 
    sleep 5s
done