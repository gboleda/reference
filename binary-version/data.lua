--[[
******* the following code is to prepare toy training *******
******* and validation data                           *******
******* we will later need to have two functions, one *******
******* for toy and one for real data                 *******
--]]

function generate_toy_data(training_set_size,validation_set_size,image_set_size,
			   t_input_size,v_input_size)

   -- word query tensor: nxe, where n is the number of items in the
   -- relevant set (training, development, testing) and e the
   -- dimensionality of word embeddings
   local training_word_query_list=torch.randn(training_set_size,t_input_size)
   local validation_word_query_list=torch.randn(validation_set_size,t_input_size)

   -- image set table: 
   -- a nxs table of i-dimensional tensors, where s is the maximum number of images in
   -- a set and i is the dimensionality of the image embeddings
   -- NB1: this is a table, to make it easier to read the separate image vectors
   -- into the network
   -- NB2: image sets in this tensor are assumed to be associated to the word
   -- in the same position in the word query tensor
   -- NB3: assuming that all image sets have same (maximum) size, with
   -- smaller sets padded with 0 vectors
   local training_image_set_list={}
   for i=1,training_set_size do
      training_image_set_list[i]={}
      for j=1,image_set_size do
	 table.insert(training_image_set_list[i],torch.randn(v_input_size))
      end
   end
   local validation_image_set_list={}
   for i=1,validation_set_size do
      validation_image_set_list[i]={}
      for j=1,image_set_size do
	 table.insert(validation_image_set_list[i],torch.randn(v_input_size))
      end
   end

   -- response tensor: a tensor of 1s and 0s for the responses to the
   -- question "is query in image set?"
   -- NB: again, order with respect to the previous two tensors should be
   -- meaningful
   local training_response_list=torch.Tensor(training_set_size):apply(function() return torch.bernoulli() end)
   local validation_response_list=torch.Tensor(validation_set_size):apply(function() return torch.bernoulli() end)

   return training_word_query_list,validation_word_query_list,training_image_set_list,validation_image_set_list,training_response_list,validation_response_list

end

 
