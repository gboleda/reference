# datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny-nodeviants'
# # BEGIN TINY5 DATA
# prefixoutput=tiny5-nodeviants
# prefixinput=stimuli-tiny
# trsetsize=5
# vsetsize=5
# tesetsize=5
# testfile=$datadir/stimuli-tiny.test
# # END TINY5 DATA

# # BEGIN TINY DATA
# prefixoutput=tiny-nodeviants
# prefixinput=stimuli
# trsetsize=40
# vsetsize=40
# tesetsize=30
# # END TINY DATA

# model=ff_ref_sim_sum
model=max_margin_bl

# datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny'
# prefixoutput=tiny-deviants
datadir='/Users/gboleda/Desktop/data/reference/new-exp3'

modelfile=$prefixoutput-$model.model
prefixinput=stimuli.deviant
trsetsize=50
vsetsize=50
tesetsize=50

margin=0.1
max_epochs=1
min_epochs=1
mini_batch_size=1
debug=0
outfile=$prefixoutput-$model.preds
modelfile=$prefixoutput-$model.model
testfile=$datadir/$prefixinput.test

# # 1. We create the model
# th main.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --test_set_size $tesetsize --image_set_size 5 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixinput --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size --save_model_to_file $modelfile

# 2. We apply it to the test set

file=stimuli.test
# filesize=10000
# predoutfile=$file-$prefixoutput.preds
# th test_with_trained_file.lua --model_file $prefixoutput.model --model $model --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --image_set_size $image_set_size --normalize_embeddings 1 --test_file $datadir/$file --test_set_size $filesize --output_guesses_file $predoutfile --modifier_mode $modifier_mode
