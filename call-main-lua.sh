date

# WHITE CAT

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

# BEGIN TINY DATA
datadir='/Users/gboleda/Desktop/love-project/data/bw-tiny'
prefixoutput=bw-tiny
trsetsize=34
vsetsize=31
tesetsize=35
# END TINY DATA
# # BEGIN REAL DATA
# datadir='/Users/gboleda/Desktop/love-project/data/exp2-4a'
# trsetsize=27981
# vsetsize=3503
# tesetsize=6978
# # END REAL DATA

# MODEL PARAMETERS
prefixfortraining=stimuli-nodevs # we train the model on non-deviant data only
max_epochs=1
min_epochs=1
model=max_margin_bl
image_set_size=5
margin=0.5
mini_batch_size=1 # consider 10, 20, 100 future
lr_decay=0.0001 # values for future: 0.001 0.0001
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9
learning_rate=0.09
ref_size=300
modifier_mode=1
modelfile=$prefixoutput-max-margin.model

echo datadir: $datadir

th main.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --test_set_size $tesetsize --image_set_size $image_set_size --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixfortraining --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size $ref_size --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --save_model_to_file $modelfile --modifier_mode $modifier_mode

date

