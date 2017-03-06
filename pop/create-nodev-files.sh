# to be called from within the data directory where the files are
# example: /Users/gboleda/Documents/github/reference/create-nodev-files.sh

prefix=stimuli
for i in train valid test
do
    awk '$2>0' $prefix.$i > $prefix-nodevs.$i
done
awk '$2<1' $prefix.train > $prefix-devs.train
