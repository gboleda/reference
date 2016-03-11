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
   -- debug
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
   -- debug
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

