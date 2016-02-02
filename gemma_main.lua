-- this is getting a bit cumbersome -- maybe we want to use some other structure:
training_word_query_list,
validation_word_query_list,
training_image_set_list,
validation_image_set_list,
training_index_list,
validation_index_list,
training_set_size,
validation_set_size,
image_set_size,
t_input_size,
v_input_size=
   load_data(opt.word_lexicon_file,opt.image_dataset_file,opt.stimuli_prefix)

