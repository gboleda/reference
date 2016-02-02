-- NB: model names should be consistent with those in the models.txt file

-- feed-forward network with reference layer
function ff_reference(t_input_size,v_input_size,image_set_size,reference_size)

   local inputs = {}
   local reference_vectors = {}

   -- text input
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query = nn.Linear(t_input_size, reference_size)(curr_input)

   -- reshaping the reference_size-dimensional text vector into 
   -- a 1xreference_size-dimensional vector for the multiplication below
   local query_matrix=nn.View(1,-1)(query)

   -- visual vectors
   for i=1,image_set_size do
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local reference_vector = nn.Linear(v_input_size,reference_size)(curr_input)
      if i>1 then -- share parameters of each reference vector
	 reference_vector.data.module:share(reference_vectors[1].data.module,'weight','bias','gradWeight','gradBias')
      end
      table.insert(reference_vectors, reference_vector)
   end

   -- reshaping the table into a matrix with a reference
   -- vector per row
   local all_reference_values=nn.JoinTable(1)(reference_vectors)
   local reference_matrix=nn.View(#reference_vectors,-1)(all_reference_values)

   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector=nn.MM(false,true)({query_matrix,reference_matrix})

   -- transforming the dot products into probabilities 
   -- via a softmax
   local importance_weights = nn.SoftMax()(dot_vector)

   -- now we use the probabilities as weights to sum
   -- the reference vectors
   local weighted_reference=nn.MM(false,false)({importance_weights,reference_matrix})

   -- finally, we formulate a binary classification task using the
   -- concatenation of the weighted reference vector and the query vector
   -- as input for a binary logistic regression
   -- (note that here we are actually just adding two scalars although we use
   -- CAddTable)
   local weighted_reference_score = nn.Linear(reference_size,1)(weighted_reference)
   local query_score = nn.Linear(reference_size,1)(query)
   local untransformed_model_guess = nn.CAddTable()({weighted_reference_score, query_score})
   local model_guess = nn.Sigmoid()(untransformed_model_guess) 

   -- wrapping up, here is our model...
   return nn.gModule(inputs,{model_guess})
end
