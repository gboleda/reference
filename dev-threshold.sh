# get thresholds

# 1. Get threshold distribution from deviant training data
# 12019 stimuli-devs.train

# t0
# awk '$1==0' stimuli-devs.train-new-exp3-max_margin_bl-0.5-300-0.09-0.0001-0.09-20-1.model.preds.andgold| perl -ane '$max=$F[2]; print "$max\n"'>max-thresholds
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

infile=stimuli.valid-new-exp3-max_margin_bl-0.5-300-0.09-0.0001-0.09-20-1.model.preds.andgold
ofile=threshold-optimization
rm $ofile
for t0 in -0.1 0 0.1 0.2 0.3 0.4
do
    for t1 in 0.03 0.06 0.09 0.12 0.15 0.18
    do
	perl accuracy-computation-with-deviants-for-max-margin.pl $t0 $t1 < $infile >> $ofile 2>> verbose-threshold-optimization
    done
done

