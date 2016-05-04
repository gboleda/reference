# EVALUATE MAX MARGIN BASELINE
# gbt, May 2016

date

# --- 1. SETTING PARAMETERS COMMON TO EVERYTHING

resultsdir='results'

# --- 2. SETTING TASK AND DATA-RELATED VARIABLES: WHITE CAT, OR BIOLOGIST WOMAN?

# WHITE CAT
# BEGIN TINY DATA
# datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny'
# prefixoutput=tiny-deviants
# END TINY DATA
# BEGIN REAL DATA
datadir='/Users/gboleda/Desktop/love-project/data/new-exp3'
prefix=tc-new-exp3-max-margin
# END REAL DATA

echo datadir: $datadir

# FILES
modelfile=$prefix.model
echo "model file: $modelfile"

# --- 3. Recording the gold indices in the predictions files

for file in stimuli-devs.trainvalid stimuli.valid stimuli.test
do
    predfile=$file-$prefix.preds
    predandgoldfile=$file-$prefix.predandgold
    cut -f2 $datadir/$file > $file.gold
    paste -d' ' $file.gold $predfile > $predandgoldfile
    echo "gold and predictions file: $predandgoldfile"
    rm $file.gold
done

# ---  4. Getting / optimizing the thresholds

# OLD
# awk '$1==0' $predandgoldfile | perl -ane 'END {$avg=$maxsum/$count; print "AVERAGE MAX: $avg\n"} $count++; $max=$F[2]; $maxsum=$maxsum+$max;'
# awk '$1==-1' $predandgoldfile | perl -ane 'BEGIN {$maxdiffsum=0} END {$avg=$maxdiffsum/$count; print "AVERAGE DIFF: $avg\n"} use List::Util max; $count++; $max=$F[2]; shift @F; shift @F; shift @F; @SORTED=sort(@F); $max2=$SORTED[-2];$maxdiff=$max-$max2; $maxdiffsum=$maxdiffsum+$maxdiff'
# # ==> obtained:
# # AVERAGE MAX: 0.133836451621809
# # AVERAGE DIFF: 0.104982467192016
# threshold0=0.133836451621809
# threshold1=0.104982467192016

# 1. Get threshold distribution from deviant training data
# stimuli-devs.train
# t0
# awk '$1==0' stimuli-devs.train-$prefix.predandgold | perl -ane '$max=$F[2]; print "$max\n"'>max-thresholds
# distribution of the max
# > a <- read.csv('max-thresholds',header=FALSE)
# > summary(a)
#        V1          
#  Min.   :-0.23420  
#  1st Qu.: 0.06934  
#  Median : 0.12955  
#  Mean   : 0.13377  
#  3rd Qu.: 0.19253  
#  Max.   : 0.54038 


# t1
# awk '$1==-1' stimuli-devs.train-new-exp3-max_margin_bl-0.5-300-0.09-0.0001-0.09-20-1.model.preds.andgold| perl -ane 'use List::Util max; $count++; $max=$F[2]; shift @F; shift @F; shift @F; @SORTED=sort(@F); $max2=$SORTED[-2];$maxdiff=$max-$max2; print "$maxdiff\n"' > maxdiff-thresholds
# > a <- read.csv('maxdiff-thresholds',header=FALSE)
# > summary(a)
#        V1         
#  Min.   :0.00000  
#  1st Qu.:0.03260  
#  Median :0.07419  
#  Mean   :0.10384  
#  3rd Qu.:0.14722  
#  Max.   :0.69577


# optimizing threshold on validation data

infile=stimuli.valid-$prefix.predandgold
ofile=threshold-optimization
rm $ofile
for t0 in -0.1 0 0.1 0.2 0.3 0.4
do
    for t1 in 0.03 0.06 0.09 0.12 0.15 0.18
    do
	perl accuracy-computation-with-deviants-for-max-margin.pl $t0 $t1 < $infile >> $ofile 2>> verbose-threshold-optimization
    done
done




# 7. We obtain accuracy on the test file and print adjusted predictions (predictions taking into account deviance)

# adjpredoutfile=$file-$prefixoutput-$model.adjustedpreds
# # # # to do: change perl script such that it outputs 0 and -1 as 1) gold, 2) predictions for max margin baseline (currently it outputs 6)
# perl accuracy-computation-with-deviants-for-max-margin.pl $threshold0 $threshold1 < $predandgoldfile > $adjpredoutfile
# echo "perl accuracy-computation-with-deviants-for-max-margin.pl $threshold0 $threshold1 < $predandgoldfile > $adjpredoutfile"
# echo "adjusted predictions saved in file $adjpredoutfile"


# 8. Now we redo the last few steps for a couple more files, to check stuff

   #  3022 stimuli-devs.test
   # 12019 stimuli-devs.train
   # 13516 stimuli-devs.trainvalid
   #  1497 stimuli-devs.valid
   #  6978 stimuli-nodevs.test
   # 27981 stimuli-nodevs.train
   #  3503 stimuli-nodevs.valid
   # 10000 stimuli.test
   # 40000 stimuli.train
   #  5000 stimuli.valid

# file=$prefix.valid
# filesize=5000
# predoutfile=$file-$modelfile.preds
# th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile


# VALIDATION

# NO DEVIANTS
# file=$prefix-nodevs.valid
# filesize=3503
# predoutfile=$file-$modelfile.preds
# th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile

# predandgoldfile=$predoutfile.andgold
# cut -f2 $datadir/$file > $file.gold
# paste -d' ' $file.gold $predoutfile > $predandgoldfile
# echo "gold and predictions file: $predandgoldfile"
# perl -ane 'END {$avg=$maxsum/$count; print "AVERAGE MAX: $avg\n"} $count++; $max=$F[2]; $maxsum=$maxsum+$max;' $predandgoldfile
# perl -ane 'BEGIN {$maxdiffsum=0} END {$avg=$maxdiffsum/$count; print "AVERAGE DIFF: $avg\n"} use List::Util max; $count++; $max=$F[2]; shift @F; shift @F; shift @F; @SORTED=sort(@F); $max2=$SORTED[-2];$maxdiff=$max-$max2; $maxdiffsum=$maxdiffsum+$maxdiff' $predandgoldfile
# ==> obtained:

# AND We remove unnecessary files
# rm *.gold

# file=stimuli.valid
# filesize=5000
# predoutfile=$file-$modelfile.preds
# # th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile

# # stimuli.valid-new-exp3-max_margin_bl-0.5-300-0.09-0.0001-0.09-20-1.model.preds

# predandgoldfile=$predoutfile.andgold
# cut -f2 $datadir/$file > $file.gold
# paste -d' ' $file.gold $predoutfile > $predandgoldfile
# echo "gold and predictions file: $predandgoldfile"
# perl -ane 'END {$avg=$maxsum/$count; print "AVERAGE MAX: $avg\n"} $count++; $max=$F[2]; $maxsum=$maxsum+$max;' $predandgoldfile
# perl -ane 'BEGIN {$maxdiffsum=0} END {$avg=$maxdiffsum/$count; print "AVERAGE DIFF: $avg\n"} use List::Util max; $count++; $max=$F[2]; shift @F; shift @F; shift @F; @SORTED=sort(@F); $max2=$SORTED[-2];$maxdiff=$max-$max2; $maxdiffsum=$maxdiffsum+$maxdiff' $predandgoldfile
