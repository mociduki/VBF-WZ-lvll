for i in 0
do
    name=WZ_HyperOpt_BDT
    echo "bash launch_BDT.sh |& tee $PWD/logs/$name.LOG" \
	| qsub -v "NAME=$name,DATA_PREFIX=$data_prefix,TYPE=$type" \
	-N $name \
	-d $PWD \
	-l nice=0 \
	-j oe \
	-o $PWD/logs \
	-l nodes=1:ppn=2 \
	-q atlas 
    sleep 5s
done
