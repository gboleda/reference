date

image_set_size=5
model=max_margin_bl

datadir='/Users/gboleda/Desktop/love-project/data/new-exp3'
echo datadir: $datadir

prefixoutput=new-exp3
modelfile=$prefixoutput-$model.model
prefixinput=stimuli-nodevs
trsetsize=27981
vsetsize=3503
tesetsize=6978

margin=0.5
max_epochs=8 # Paper: 1 ... 14* ... 50 (for mini-batch sizes >1, epochs up to mini-batch size x 15)             
min_epochs=8
mini_batch_size=1 # consider 10, 20, 100 future                                                                  
lr_decay=0.0001 # values for future: 0.001 0.0001                                                                
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9                                                     
learning_rate=0.09
ref_size=300                                                                                    
#debug=0
modelfile=$prefixoutput-$model-$margin-$ref_size-$learning_rate-$lr_decay-$momentum-$max_epochs-$mini_batch_size.model


outfile=$prefixoutput-$model.preds
testfile=$datadir/$prefixinput.test

# 1. We create the model
th main.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --test_set_size $tesetsize --image_set_size $image_set_size --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixinput --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size $ref_size --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --save_model_to_file $modelfile
