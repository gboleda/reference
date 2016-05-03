# to be called from within the data directory where the files are
prefix=stimuli
for i in train valid test
do
    awk '$2>0' $prefix.$i > $prefix-nodevs.$i
    awk '$2<1' $prefix.$i > $prefix-devs.$i
done
cat $prefix-devs.train $prefix-devs.valid > $prefix-devs.trainvalid
