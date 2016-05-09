# create results for max margin baseline

# CALL THIS SCRIPT FROM WITHIN THE LUA CODE FOLDER (GITHUB)

date

resultsdir='results'

# 0. We obtain the relevant devs/nodevs files

# execute create-nodev-files.sh from data folder
# gbt: to do: streamline create-nodev-files.sh so it creates only the necessary files

# 1. We create the model using nodeviant training + validation data

# BEGIN TINY DATA
# datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny'
# prefixoutput=tiny-deviants
# prefixinput=stimuli.deviant *** WATCH OUT
# trsetsize=40
# vsetsize=42
# tesetsize=39
# END TINY DATA

# BEGIN REAL DATA
datadir='/Users/gboleda/Desktop/love-project/data/new-exp3'
prefix=stimuli
prefixoutput=new-exp3
prefixinput=$prefix-nodevs
trsetsize=27981
vsetsize=3503
tesetsize=6978
# END REAL DATA

echo datadir: $datadir

# MODEL PARAMETERS
model=max_margin_bl
image_set_size=5
margin=0.5
max_epochs=6 # Paper: 1 ... 14* ... 50 (for mini-batch sizes >1, epochs up to mini-batch size x 15)
min_epochs=6
mini_batch_size=1 # consider 10, 20, 100 future
lr_decay=0.0001 # values for future: 0.001 0.0001
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9
learning_rate=0.09
ref_size=300
modelfile=$prefixoutput-$model-$margin-$ref_size-$learning_rate-$lr_decay-$momentum-$max_epochs-$mini_batch_size.model
# echo $modelfile

modeltrainingprefixinput=$prefixinput-nodevs

# note: the following has not been re-tested
echo "th main.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --test_set_size $tesetsize --image_set_size $image_set_size --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixinput --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size $ref_size --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --save_model_to_file $modelfile"

th main.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --test_set_size $tesetsize --image_set_size $image_set_size --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixinput --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size $ref_size --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --save_model_to_file $modelfile

# 2. We apply the model to the deviant training + validation data to obtain the predictions

# BEGIN TINY DATA
# filesize=18
# END TINY DATA
# BEGIN REAL DATA
filesize=13516
# END REAL DATA

file=$prefix-devs.trainvalid
predoutfile=$file-$modelfile.preds

echo "th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile"

# to do: see if accuracy with "8 model" is the same
th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile

# # 3. We record the gold indices in the predictions file

# predandgoldfile=$predoutfile.andgold
# cut -f2 $datadir/$file > $file.gold
# paste -d' ' $file.gold $predoutfile > $predandgoldfile
# echo "gold and predictions file: $predandgoldfile"

# # # 4. We get the thresholds

# # # 4.1. For 0 deviants
# awk '$1==0' $predandgoldfile | perl -ane 'END {$avg=$maxsum/$count; print "AVERAGE MAX: $avg\n"} $count++; $max=$F[2]; $maxsum=$maxsum+$max;'
# # 4.2. For -1 deviants
# awk '$1==-1' $predandgoldfile | perl -ane 'BEGIN {$maxdiffsum=0} END {$avg=$maxdiffsum/$count; print "AVERAGE DIFF: $avg\n"} use List::Util max; $count++; $max=$F[2]; shift @F; shift @F; shift @F; @SORTED=sort(@F); $max2=$SORTED[-2];$maxdiff=$max-$max2; $maxdiffsum=$maxdiffsum+$maxdiff'
# # ==> obtained:
# # AVERAGE MAX: 0.133836451621809
# # AVERAGE DIFF: 0.104982467192016
# threshold0=0.133836451621809
# threshold1=0.104982467192016

# # 5. We apply the model to the test file to obtain the predictions

file=$prefix.test
# BEGIN TINY DATA
# filesize=50
# END TINY DATA
# BEGIN REAL DATA
filesize=10000
# END REAL DATA

predoutfile=$file-$modelfile.preds
echo "th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile"
th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile

# 6. We record the gold indices in the test predictions file

# predandgoldfile=$predoutfile.andgold
# cut -f2 $datadir/$file > $file.gold
# paste -d' ' $file.gold $predoutfile > $predandgoldfile

# echo "predandgoldfile=$predoutfile.andgold
# cut -f2 $datadir/$file > $file.gold
# paste -d' ' $file.gold $predoutfile > $predandgoldfile"

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

file=$prefix.valid
filesize=5000
predoutfile=$file-$modelfile.preds
th test_with_trained_file.lua --model_file $modelfile --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile


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
