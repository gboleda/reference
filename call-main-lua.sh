date

margin=0.1
max_epochs=1
min_epochs=1
mini_batch_size=1

datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny'
th main.lua --model ff_ref_with_summary --training_set_size 50 --validation_set_size 50 --test_set_size 50 --image_set_size 5 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli.deviant --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size --save_model_to_file tiny-ff_ref_with_summary.model


# -- tiny-nodeviants dataset

# datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny-nodeviants'

# th main.lua --model ff_ref --training_set_size 40 --validation_set_size 40 --test_set_size 30 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size

# th main.lua --model ff_ref_deviance --training_set_size 5 --validation_set_size 5 --test_set_size 5 --image_set_size 5 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli-tiny --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size --save_model_to_file tiny-ff_ref_deviance.model

# th main.lua --model max_margin_bl --margin $margin --training_set_size 40 --validation_set_size 40 --test_set_size 30 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size

# th main.lua --model max_margin_bl --margin $margin --training_set_size 5 --validation_set_size 5 --test_set_size 5 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli-tiny --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size

# # -- 10K-3K-5K, no deviants
# datadir='/Users/gboleda/Desktop/data/reference/new-exp3a'
# th main.lua --model max_margin_bl --margin $margin --training_set_size 10000 --validation_set_size 3000 --test_set_size 5000 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size --save_model_to_file new-exp3a-max-margin-bl-margin01.model

# -- new-exp3, training on no deviant portion
# datadir='/Users/gboleda/Desktop/data/reference/new-exp3'
# th main.lua --model max_margin_bl --margin $margin --training_set_size 27981 --validation_set_size 3503 --test_set_size 6978 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli-nodevs --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size --save_model_to_file new-exp3-max-margin-bl.model

date
