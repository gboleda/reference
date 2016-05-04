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
#datadir='/Users/gboleda/Desktop/love-project/data/new-exp3'
#prefix=tc-new-exp3-max-margin
# END REAL DATA

# BIOLOGIST WOMAN
datadir='/home/gemma.boledatorrent/white-cat-experiment/conll-data/exp2-4a'
prefix=bw-exp2-4a-max-margin

echo datadir: $datadir

# --- 3. Recording the gold indices in the predictions files

for file in stimuli-devs.train stimuli.valid stimuli.test
do
    predfile=$file-$prefix.preds
    predandgoldfile=$file-$prefix.predandgold
    cut -f2 $datadir/$file > $file.gold
    paste -d' ' $file.gold $predfile > $predandgoldfile
    echo "gold and predictions file: $predandgoldfile"
    rm $file.gold
done

# ---  4. Getting / optimizing the thresholds

# SEE SCRIPT dev-threshold.sh

# OLD
# awk '$1==0' $predandgoldfile | perl -ane 'END {$avg=$maxsum/$count; print "AVERAGE MAX: $avg\n"} $count++; $max=$F[2]; $maxsum=$maxsum+$max;'
# awk '$1==-1' $predandgoldfile | perl -ane 'BEGIN {$maxdiffsum=0} END {$avg=$maxdiffsum/$count; print "AVERAGE DIFF: $avg\n"} use List::Util max; $count++; $max=$F[2]; shift @F; shift @F; shift @F; @SORTED=sort(@F); $max2=$SORTED[-2];$maxdiff=$max-$max2; $maxdiffsum=$maxdiffsum+$maxdiff'
# # ==> obtained:
# # AVERAGE MAX: 0.133836451621809
# # AVERAGE DIFF: 0.104982467192016

# # TC - new-exp3
# threshold0=0.1
# threshold1=0.03

# BW
threshold0=0.4
threshold1=0.07

# --- 5. Obtain accuracy on the test file and print adjusted predictions (predictions taking into account deviance)

adjpredoutfile=stimuli.test-$prefix.adjpreds
perl accuracy-computation-with-deviants-for-max-margin.pl $threshold0 $threshold1 yes < stimuli.test-$prefix.predandgold > $adjpredoutfile
echo "adjusted predictions saved in file $adjpredoutfile"

date
