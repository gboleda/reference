date

# BINDING

# # BEGIN TINY DATA
# datadir='/Users/gboleda/Desktop/love-project/data/binding/exp-tiny'
# trsetsize=200
# vsetsize=50
# tesetsize=100
# prefixoutput=binding-exp-tiny
# # END TINY DATA

# BEGIN LARGE DATA
datadir='/Users/gboleda/Desktop/love-project/data/binding/exp1-large'
trsetsize=4000
vsetsize=5000
tesetsize=10000
prefixoutput=binding-exp1-large
# END LARGE DATA

# MODEL PARAMETERS
prefixfortraining=stimuli
max_epochs=3
min_epochs=1
model=ff
mini_batch_size=1 # consider 10, 20, 100 future
lr_decay=0.0001 # values for future: 0.001 0.0001
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9
learning_rate=0.09
input_sequence_cardinality=6
hidden_count=1 # number of hidden layers [1]
ff_nonlinearity=none # nonlinear transformation of hidden layers (options: none (default), sigmoid, relu, tanh) [none]
modelprefix=$prefixoutput-ff-$hidden_count-hiddenlayers
modelfile=$modelprefix.bin
test_file=stimuli.test
output_guesses_file=$prefixoutput-$test_file.guesses

echo datadir: $datadir

echo command: "th lent-and-returned-train.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixfortraining --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --save_model_to_file $modelfile --input_sequence_cardinality $input_sequence_cardinality --hidden_count $hidden_count --ff_nonlinearity $ff_nonlinearity --max_validation_lull 5"

th lent-and-returned-train.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixfortraining --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --save_model_to_file $modelfile --input_sequence_cardinality $input_sequence_cardinality --hidden_count $hidden_count --ff_nonlinearity $ff_nonlinearity --max_validation_lull 5

#th lent-and-returned-test.lua --test_set_size $tesetsize --test_file $datadir/$test_file --input_sequence_cardinality $input_sequence_cardinality --model_file $modelfile --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --normalize_embeddings 1 --output_guesses_file $output_guesses_file

date

# best parameters from marco
# th lent-and-returned-train.lua --protocol_prefix  /home/sebastian.pado/binding/exp1-large/stimuli --word_embedding_file 
# /home/sebastian.pado/binding/exp1-large/word.dm  --image_embedding_file /home/sebastian.pado/binding/exp1-large/image.dm
#  --normalize_embeddings 1 --input_sequence_cardinality 6 --training_set_size 40000 --validation_set_size 5000 --multimod
# al_size 300 --mini_batch_size 1 --learning_rate 0.09 --momentum 0.09 --learning_rate_decay 0.0001 --save_model_to_file t
# olerant-exp1-large.bin --new_mass_aggregation_method sum --max_validation_lull 5
