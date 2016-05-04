# TRAIN AND APPLY MODEL (FOCUSING ON MAX MARGIN BASELINE)
# gbt, May 2016
# NOTE: call from within the Lua code folder

date

# --- 0. OBTAIN THE RELEVANT DEVS/NODEVS FILES

# execute create-nodev-files.sh from within data folder

# --- 1. SETTING PARAMETERS COMMON TO EVERYTHING

max_epochs=6
min_epochs=6
model=max_margin_bl
image_set_size=5
margin=0.5
mini_batch_size=1 # consider 10, 20, 100 future
lr_decay=0.0001 # values for future: 0.001 0.0001
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9
learning_rate=0.09
ref_size=300

# --- 2. SETTING TASK AND DATA-RELATED VARIABLES: WHITE CAT, OR BIOLOGIST WOMAN?

# WHITE CAT

# MODEL PARAMETERS
# modifier_mode=0

# BEGIN TINY DATA
# datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny'
# trsetsize=40
# vsetsize=42
# tesetsize=39
# END TINY DATA
# # BEGIN REAL DATA
# datadir='/Users/gboleda/Desktop/love-project/data/new-exp3'
# prefixinput=$stimuli-nodevs
# trsetsize=27981
# vsetsize=3503
# tesetsize=6978
# # END REAL DATA

# BIOLOGIST WOMAN

# MODEL PARAMETERS
modifier_mode=1

# # BEGIN TINY DATA
# datadir='/Users/gboleda/Desktop/love-project/data/bw-tiny'
# prefixoutput=bw-tiny
# trsetsize=34
# vsetsize=31
# tesetsize=35
# # END TINY DATA
# # BEGIN REAL DATA
# datadir='/Users/gboleda/Desktop/love-project/data/exp2-4a'
datadir='/home/boledatorrent/reference/data/exp2-4a'
prefixoutput=bw-exp2-4a-max-margin
  # 12124 stimuli-devs.train
  #  6906 stimuli-nodevs.test
  # 27876 stimuli-nodevs.train
  #  3448 stimuli-nodevs.valid
  # 10000 stimuli.test
  # 40000 stimuli.train
  #  5000 stimuli.valid
# # END REAL DATA

# --- 3. TRAINING THE MODEL

# PARAMETERS FOR TRAINING
prefixfortraining=stimuli-nodevs # we train the model on non-deviant data only
modelfile=
# # BEGIN TINY DATA
# trsetsize=34
# vsetsize=31
# tesetsize=35
# # END TINY DATA
# BEGIN REAL DATA
trsetsize=27876
vsetsize=3448
tesetsize=6906
# END REAL DATA

th main.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --test_set_size $tesetsize --image_set_size $image_set_size --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixfortraining --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size $ref_size --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --save_model_to_file $prefixoutput.model --modifier_mode $modifier_mode

date

# --- 4. APPLYING THE MODEL

# training file with deviants, to obtain the range for the thresholds
file=stimuli-devs.train
filesize=12124
predoutfile=$file-$prefixoutput.preds
th test_with_trained_file.lua --model_file $prefixoutput.model --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile --modifier_mode $modifier_mode

# validation file, to optimize the threshold
file=stimuli.valid
filesize=5000
predoutfile=$file-$prefixoutput.preds
th test_with_trained_file.lua --model_file $prefixoutput.model --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile --modifier_mode $modifier_mode

# test file, to evaluate the resulting model
file=stimuli.test
filesize=10000
predoutfile=$file-$prefixoutput.preds
th test_with_trained_file.lua --model_file $prefixoutput.model --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile --modifier_mode $modifier_mode

date

echo 'done!'
