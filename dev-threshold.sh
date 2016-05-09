# get thresholds

#prefix=tc-new-exp3-max-margin
prefix=bw-exp2-4a-max-margin

# 1. Get threshold distribution from DEVIANT TRAINING data

awk '$1==0' stimuli-devs.train-$prefix.predandgold | perl -ane '$max=$F[2]; print "$max\n"'>max-thresholds
awk '$1==-1'  stimuli-devs.train-$prefix.predandgold| perl -ane 'use List::Util max; $count++; $max=$F[2]; shift @F; shift @F; shift @F; @SORTED=sort(@F); $max2=$SORTED[-2];$maxdiff=$max-$max2; print "$maxdiff\n"' > maxdiff-thresholds

# --- for tc-new-exp3-max-margin
# t0: distribution of the max
# a <- read.csv('max-thresholds',header=FALSE)
# summary(a)
#        V1          
#  Min.   :-0.20928  
#  1st Qu.: 0.06992  
#  Median : 0.13283  
#  Mean   : 0.13696  
#  3rd Qu.: 0.19701  
#  Max.   : 0.56617

# t1: distribution of the maxdiff
# a <- read.csv('maxdiff-thresholds',header=FALSE)
# summary(a)
#       V1         
# Min.   :0.00000  
# 1st Qu.:0.03262  
# Median :0.07996  
# Mean   :0.10788  
# 3rd Qu.:0.15119  
# Max.   :0.71508

# --- bw-exp2-4a-max-margin
# > a <- read.csv('max-thresholds',header=FALSE)
# > summary(a)
#        V1         
#  Min.   :-0.1746  
#  1st Qu.: 0.4026  
#  Median : 0.5379  
#  Mean   : 0.5096  
#  3rd Qu.: 0.6481  
#  Max.   : 1.0675  
# > a <- read.csv('maxdiff-thresholds',header=FALSE)
# > summary(a)
#        V1         
#  Min.   :0.00000  
#  1st Qu.:0.03058  
#  Median :0.06494  
#  Mean   :0.08204  
#  3rd Qu.:0.11805  
#  Max.   :0.43233

# 2. optimizing threshold on VALIDATION data

infile=stimuli.valid-$prefix.predandgold
rm threshold-optimization
for t0 in -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    for t1 in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.15 0.18 0.21
    do
	perl accuracy-computation-with-deviants-for-max-margin.pl $t0 $t1 no < $infile >> threshold-optimization
    done
done

sort -nrk3 threshold-optimization |head

# for tc-new-exp3-max-margin
# 0.1 0.03 0.6506
# for bw-exp2-4a-max-margin
# 0.4 0.07 0.6568
