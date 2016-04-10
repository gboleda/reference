-- feed-forward network with reference layer OLD: does not handle modifiers!
function ff_reference_old(t_inp_size,v_inp_size,img_set_size,ref_size)

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

-- max margin baseline UNSTABLE
function max_margin_baseline_model(t_inp_size,v_inp_size,ref_size)
   local inputs = {}

   -- inputs
   local ling = nn.Identity()()
   table.insert(inputs,ling)
   local target_i = nn.Identity()()
   table.insert(inputs,target_i)
   local confounder_i = nn.Identity()()
   table.insert(inputs,confounder_i)

   -- mappings
   local query = nn.LinearNB(t_inp_size, ref_size)(ling)
   local target_ref = nn.LinearNB(v_inp_size,ref_size)(target_i)
   local confounder_ref = nn.LinearNB(v_inp_size,ref_size)(confounder_i)
   confounder_ref.data.module:share(target_ref.data.module,'weight','bias','gradWeight','gradBias')
   
   -- dot products
   local right = nn.DotProduct()({query, target_ref})
   local wrong = nn.DotProduct()({query, confounder_ref})
   
   -- wrapping up, here is our model...
   return nn.gModule(inputs,{right,wrong}) -- output of the model: dot products

end

