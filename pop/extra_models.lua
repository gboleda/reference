-- feed-forward network with reference layer and summed reference vectors
-- to predict anomalies
function ff_reference_with_reference_summary(t_inp_size,v_inp_size,img_set_size,ref_size)

   local inputs = {}
   local reference_vectors = {}

   -- text input
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query = nn.LinearNB(t_inp_size, ref_size)(curr_input)

   -- reshaping the ref_size-dimensional text vector into 
   -- a 1xref_size-dimensional vector for the multiplication below
   -- NB: setNumInputDims method warns nn that it might get a
   -- two-dimensional object, in which case it has to treat it as a
   -- batch of 1-dimensional objects
   local query_matrix=nn.View(1,-1):setNumInputDims(1)(query)

   -- visual vectors
   for i=1,img_set_size do

      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local reference_vector = nn.LinearNB(v_inp_size,ref_size)(curr_input)
      if i>1 then -- share parameters of each reference vector
	 reference_vector.data.module:share(reference_vectors[1].data.module,'weight','bias','gradWeight','gradBias')
     end
      table.insert(reference_vectors, reference_vector)
   end

   -- reshaping the table into a matrix with a reference
   -- vector per row
   local all_reference_values=nn.JoinTable(1,1)(reference_vectors) -- second argument to JoinTable tells it that 
                                                                   -- the expected inputs in the table are one-dimensional (the
                                                                   --  reference vectors), necessary not to confuse 
                                                                   -- it when batches are passes
   local reference_matrix=nn.View(#reference_vectors,-1):setNumInputDims(1)(all_reference_values) -- again, note 
                                                                                                  -- setNumInputDims for
   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector_split=nn.MM(false,true)({query_matrix,reference_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing

   -- at the same time, we will use the reference matrix to predict
   -- deviance
   local averaged_reference_vector = nn.Mean(1,2)(reference_matrix)
   local deviance_value = nn.Linear(ref_size,1)(averaged_reference_vector)

   -- reshaping into 1 by 1 tensor for concatenation beloow
   local deviance_cell = nn.View(1,1)(deviance_value)
   -- concatenating
   local extended_dot_vector = nn.JoinTable(2)({dot_vector,deviance_cell})

   -- transforming the dot products into LOG probabilities via a
   -- softmax (log for compatibility with the ClassNLLCriterion)
   local relevance_distribution =  nn.LogSoftMax()(extended_dot_vector)

   -- wrapping up, here is our model...
   return nn.gModule(inputs,{relevance_distribution})
end

-- feed-forward network with reference layer and deviance layer on top of 
-- dot products
function ff_reference_with_deviance_layer(t_inp_size,v_inp_size,img_set_size,ref_size,deviance_size,nonlinearity)

   local inputs = {}
   local reference_vectors = {}

   -- text input
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query = nn.LinearNB(t_inp_size, ref_size)(curr_input)

   -- reshaping the ref_size-dimensional text vector into 
   -- a 1xref_size-dimensional vector for the multiplication below
   -- NB: setNumInputDims method warns nn that it might get a
   -- two-dimensional object, in which case it has to treat it as a
   -- batch of 1-dimensional objects
   local query_matrix=nn.View(1,-1):setNumInputDims(1)(query)

   -- visual vectors
   for i=1,img_set_size do

      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local reference_vector = nn.LinearNB(v_inp_size,ref_size)(curr_input)
      if i>1 then -- share parameters of each reference vector
	 reference_vector.data.module:share(reference_vectors[1].data.module,'weight','bias','gradWeight','gradBias')
     end
      table.insert(reference_vectors, reference_vector)
   end

   -- reshaping the table into a matrix with a reference
   -- vector per row
   local all_reference_values=nn.JoinTable(1,1)(reference_vectors) -- second argument to JoinTable tells it that 
                                                                   -- the expected inputs in the table are one-dimensional (the
                                                                   --  reference vectors), necessary not to confuse 
                                                                   -- it when batches are passes
   local reference_matrix=nn.View(#reference_vectors,-1):setNumInputDims(1)(all_reference_values) -- again, note 
                                                                                                  -- setNumInputDims for
   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector_split=nn.MM(false,true)({query_matrix,reference_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing

   -- we have a layer on top of the dot vector to reason about
   -- deviance
   local deviance_layer = nn.Linear(img_set_size,deviance_size)(dot_vector)
   local nonlinear_deviance_layer = nil
   if (nonlinearity == 'sigmoid') then
      nonlinear_deviance_layer = nn.Sigmoid()(deviance_layer)
   else
      nonlinear_deviance_layer = nn.Tanh()(deviance_layer)
   end
   local deviance_value = nn.Linear(deviance_size,1)(nonlinear_deviance_layer)

   -- reshaping into 1 by 1 tensor for concatenation beloow
   local deviance_cell = nn.View(1,1)(deviance_value)
   -- concatenating
   local extended_dot_vector = nn.JoinTable(2)({dot_vector,deviance_cell})

   -- transforming the dot products into LOG probabilities via a
   -- softmax (log for compatibility with the ClassNLLCriterion)
   local relevance_distribution =  nn.LogSoftMax()(extended_dot_vector)

   -- wrapping up, here is our model...
   return nn.gModule(inputs,{relevance_distribution})
end


-- feed-forward network with reference layer and deviance layer on top of 
-- sum of dot products
function ff_reference_with_similarity_sum_cell(t_inp_size,v_inp_size,img_set_size,ref_size,deviance_size,nonlinearity,sum_of_nonlinearities)

   local inputs = {}
   local reference_vectors = {}

   -- first, we read in the number of real images (the other are 0
   -- vectors used for padding)
   local number_of_real_images = nn.Identity()()
   table.insert(inputs,number_of_real_images)

   -- text input
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query = nn.LinearNB(t_inp_size, ref_size)(curr_input):annotate{name='query'}

   -- reshaping the ref_size-dimensional text vector into 
   -- a 1xref_size-dimensional vector for the multiplication below
   -- NB: setNumInputDims method warns nn that it might get a
   -- two-dimensional object, in which case it has to treat it as a
   -- batch of 1-dimensional objects
   local query_matrix=nn.View(1,-1):setNumInputDims(1)(query)

   -- visual vectors
   for i=1,img_set_size do

      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      reference_vector_name='reference_vector_' .. i
      local reference_vector = nn.LinearNB(v_inp_size,ref_size)(curr_input):annotate{name=reference_vector_name}
      if i>1 then -- share parameters of each reference vector
	 reference_vector.data.module:share(reference_vectors[1].data.module,'weight','bias','gradWeight','gradBias')
     end
      table.insert(reference_vectors, reference_vector)
   end

   -- reshaping the table into a matrix with a reference
   -- vector per row
   local all_reference_values=nn.JoinTable(1,1)(reference_vectors) -- second argument to JoinTable tells it that 
                                                                   -- the expected inputs in the table are one-dimensional (the
                                                                   --  reference vectors), necessary not to confuse 
                                                                   -- it when batches are passes
   local reference_matrix=nn.View(#reference_vectors,-1):setNumInputDims(1)(all_reference_values):annotate{name='reference_matrix'} -- again, note 
                                                    -- setNumInputDims for
   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector_split=nn.MM(false,true)({query_matrix,reference_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing


   -- we have a layer on top of the sum of dot vectors and the number
   -- of real images to reason about deviance 
   -- if sum_of_nonlinearities is set to sigmoid or relu, we first pass the dot_vector through a
   -- the relevant nonlinearity
   local adimensional_sum_of_dot_vectors=nil
   if (sum_of_nonlinearities == 'sigmoid') then
      local nonlinear_dot_vector = nn.Sigmoid()(dot_vector)
      adimensional_sum_of_dot_vectors=nn.Sum(1,1)(nonlinear_dot_vector) -- usual Torch silliness...
   elseif (sum_of_nonlinearities == 'relu') then
      local nonlinear_dot_vector = nn.ReLU()(dot_vector)
      adimensional_sum_of_dot_vectors=nn.Sum(1,1)(nonlinear_dot_vector) -- usual Torch silliness...
   else
      adimensional_sum_of_dot_vectors=nn.Sum(1,1)(dot_vector) -- usual Torch silliness...
   end
   local sum_of_dot_vectors=nn.View(-1,1)(adimensional_sum_of_dot_vectors)

   local deviance_layer_from_sum_of_dot_vectors = nn.Linear(1,deviance_size)(sum_of_dot_vectors)
   local deviance_layer_from_number_of_real_images = nn.Linear(1,deviance_size)(number_of_real_images)
   local deviance_layer = nn.CAddTable()({deviance_layer_from_sum_of_dot_vectors,deviance_layer_from_number_of_real_images}):annotate{name='deviance_layer'}
   local nonlinear_deviance_layer = nil
   if (nonlinearity == 'sigmoid') then
      nonlinear_deviance_layer = nn.Sigmoid()(deviance_layer)
   else
      nonlinear_deviance_layer = nn.Tanh()(deviance_layer)
   end
   local deviance_value = nn.Linear(deviance_size,1)(nonlinear_deviance_layer)

   -- reshaping into 1 by 1 tensor for concatenation beloow
   local deviance_cell = nn.View(1,1)(deviance_value)

   local extended_dot_vector = nil
   -- concatenating
   extended_dot_vector = nn.JoinTable(2)({dot_vector,deviance_cell}):annotate{name='extended_dot_vector'}

   -- transforming the dot products into LOG probabilities via a
   -- softmax (log for compatibility with the ClassNLLCriterion)
   local relevance_distribution =  nn.LogSoftMax()(extended_dot_vector)

   -- wrapping up, here is our model...
   return nn.gModule(inputs,{relevance_distribution})
end

-- feed-forward network with reference layer and deviance layer on top of 
-- sum of dot products NO INFO ON NUMBER OF INPUT VECTORS
function ff_reference_with_similarity_sum_cell_unnorm(t_inp_size,v_inp_size,img_set_size,ref_size,deviance_size,nonlinearity,sum_of_nonlinearities)

   local inputs = {}
   local reference_vectors = {}

   -- text input
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query = nn.LinearNB(t_inp_size, ref_size)(curr_input):annotate{name='query'}

   -- reshaping the ref_size-dimensional text vector into 
   -- a 1xref_size-dimensional vector for the multiplication below
   -- NB: setNumInputDims method warns nn that it might get a
   -- two-dimensional object, in which case it has to treat it as a
   -- batch of 1-dimensional objects
   local query_matrix=nn.View(1,-1):setNumInputDims(1)(query)

   -- visual vectors
   for i=1,img_set_size do

      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      reference_vector_name='reference_vector_' .. i
      local reference_vector = nn.LinearNB(v_inp_size,ref_size)(curr_input):annotate{name=reference_vector_name}
      if i>1 then -- share parameters of each reference vector
	 reference_vector.data.module:share(reference_vectors[1].data.module,'weight','bias','gradWeight','gradBias')
     end
      table.insert(reference_vectors, reference_vector)
   end

   -- reshaping the table into a matrix with a reference
   -- vector per row
   local all_reference_values=nn.JoinTable(1,1)(reference_vectors) -- second argument to JoinTable tells it that 
                                                                   -- the expected inputs in the table are one-dimensional (the
                                                                   --  reference vectors), necessary not to confuse 
                                                                   -- it when batches are passes
   local reference_matrix=nn.View(#reference_vectors,-1):setNumInputDims(1)(all_reference_values):annotate{name='reference_matrix'} -- again, note 
                                                    -- setNumInputDims for
   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector_split=nn.MM(false,true)({query_matrix,reference_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing


   -- we have a layer on top of the sum of dot vectors to reason about deviance 
   -- if sum_of_nonlinearities is set to sigmoid or relu, we first pass the dot_vector through 
   -- the relevant nonlinearity
   local adimensional_sum_of_dot_vectors=nil
   if (sum_of_nonlinearities == 'sigmoid') then
      local nonlinear_dot_vector = nn.Sigmoid()(dot_vector)
      adimensional_sum_of_dot_vectors=nn.Sum(1,1)(nonlinear_dot_vector) -- usual Torch silliness...
   elseif (sum_of_nonlinearities == 'relu') then
      local nonlinear_dot_vector = nn.ReLU()(dot_vector)
      adimensional_sum_of_dot_vectors=nn.Sum(1,1)(nonlinear_dot_vector) -- usual Torch silliness...
   else
      adimensional_sum_of_dot_vectors=nn.Sum(1,1)(dot_vector) -- usual Torch silliness...
   end
   local sum_of_dot_vectors=nn.View(-1,1)(adimensional_sum_of_dot_vectors)
   local deviance_layer = nn.Linear(1,deviance_size)(sum_of_dot_vectors):annotate{name='deviance_layer'}
   local nonlinear_deviance_layer = nil
   if (nonlinearity == 'sigmoid') then
      nonlinear_deviance_layer = nn.Sigmoid()(deviance_layer)
   else
      nonlinear_deviance_layer = nn.Tanh()(deviance_layer)
   end
   local deviance_value = nn.Linear(deviance_size,1)(nonlinear_deviance_layer)

   -- reshaping into 1 by 1 tensor for concatenation beloow
   local deviance_cell = nn.View(1,1)(deviance_value)

   local extended_dot_vector = nil
   -- concatenating
   extended_dot_vector = nn.JoinTable(2)({dot_vector,deviance_cell}):annotate{name='extended_dot_vector'}

   -- transforming the dot products into LOG probabilities via a
   -- softmax (log for compatibility with the ClassNLLCriterion)
   local relevance_distribution =  nn.LogSoftMax()(extended_dot_vector)

   -- wrapping up, here is our model...
   return nn.gModule(inputs,{relevance_distribution})
end

-- feed-forward network with reference layer DEV VERSION
function ff_reference(t_inp_size,v_inp_size,img_set_size,ref_size)

   local inputs = {}
   local reference_vectors = {}

   -- text input
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query = nn.LinearNB(t_inp_size, ref_size)(curr_input):annotate{name='query'}


   -- reshaping the ref_size-dimensional text vector into 
   -- a 1xref_size-dimensional vector for the multiplication below
   -- NB: setNumInputDims method warns nn that it might get a
   -- two-dimensional object, in which case it has to treat it as a
   -- batch of 1-dimensional objects
   local query_matrix=nn.View(1,-1):setNumInputDims(1)(query)

   -- visual vectors
   for i=1,img_set_size do

      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local reference_vector = nn.LinearNB(v_inp_size,ref_size)(curr_input)
      if i>1 then -- share parameters of each reference vector
	 reference_vector.data.module:share(reference_vectors[1].data.module,'weight','bias','gradWeight','gradBias')
     end
      table.insert(reference_vectors, reference_vector)
   end

   -- reshaping the table into a matrix with a reference
   -- vector per row
   local all_reference_values=nn.JoinTable(1,1)(reference_vectors) -- second argument to JoinTable tells it that 
                                                                   -- the expected inputs in the table are one-dimensional (the
                                                                   --  reference vectors), necessary not to confuse 
                                                                   -- it when batches are passes
   local reference_matrix=nn.View(#reference_vectors,-1):setNumInputDims(1)(all_reference_values):annotate{name='reference_matrix'} -- again, note 
                                                                                                  -- setNumInputDims for
                                                                                                  -- minibatch processing

   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector_split=nn.MM(false,true)({query_matrix,reference_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split):annotate{name='dot_vector'} -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing

   -- transforming the dot products into LOG probabilities via a
   -- softmax (log for compatibility with the ClassNLLCriterion)
   local relevance_distribution =  nn.LogSoftMax()(dot_vector)

   -- wrapping up, here is our model...
   return nn.gModule(inputs,{relevance_distribution})
end

