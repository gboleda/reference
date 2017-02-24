-- =================== --
----- PRELIMINARIES -----
-- =================== --

require('nn')
require('nngraph')
require('optim')
require('../LinearNB') -- for linear mappings without bias

print('reading the model file')
dofile('models.lua')

print('reading the data processing files')
dofile('data.lua')
dofile('data-max-margin.lua')

-- crit = nn.MarginRankingCriterion(0.1)

-- TOY TESTING WITH A MODEL FUNCTION --

-- print('--- begin ---')
local t_input_size=3
local v_input_size=2
local batchsize=4

queries=torch.Tensor(batchsize,t_input_size)
targets=torch.Tensor(batchsize,v_input_size)
q=torch.Tensor({1,3,1})
t=torch.Tensor({2,1})
confounders=torch.Tensor({{2,3},{2,5},{3,6},{4,7}})
-- confounders=torch.Tensor(batchsize,v_input_size)
for i = 1, batchsize do -- The range includes both ends.
   queries[i]=q
   targets[i]=t
end

batchinputs={queries,targets,confounders}
print('--- input ---')
print(batchinputs)
for i=1,3 do print(batchinputs[i]:dim());print(batchinputs[i]) end
-- gold=torch.Tensor(batchsize):zero()+1

local model=max_margin_baseline_model(t_input_size, v_input_size,2)
-- getting pointers to the model weights and their gradient
model_weights, model_weight_gradients = model:getParameters()
-- initializing
model_weights:ones(#model_weights) -- for inspection of model results
-- print('parameters: ' .. tostring(model_weights))
print('number of parameters in the model: ' .. model_weights:nElement())

params_ling=torch.zeros(3,2)+1
params_vis=torch.zeros(2,2)+1
qref=queries * params_ling
tref=targets * params_vis
cref=confounders * params_vis
print(qref)
print(tref)
print(cref)

d1=torch.Tensor(4)
d2=torch.Tensor(4)
for i=1,4 do
   d1[i]=qref[i]*tref[i]
   d2[i]=qref[i]*cref[i]
end
print(d1)
print(d2)

print('--- model predictions ---')
model_prediction=model:forward(batchinputs)
print(model_prediction)
print('--- indiv model pred ---')
for i=1,#model_prediction do print(model_prediction[i]) end

-- TOY TESTING WITHOUT A MODEL FUNCTION --

-- -- building {[q,q,q],[t,t,t],[c1,c2,c3]}
-- -- queries=torch.Tensor(3,t_input_size)
-- queries=torch.Tensor(3,v_input_size) -- DUMMY -- to make sizes match
-- targets=torch.Tensor(3,v_input_size)
-- confounders=torch.Tensor(3,v_input_size)
-- for i = 1, 3 do -- The range includes both ends.
--    --   queries[i]=q
--    queries[i]=t
--    targets[i]=t
-- end
-- confounders[1]=c1
-- confounders[2]=c2
-- confounders[3]=c3

-- rshp = nn.View(1,-1):setNumInputDims(1)
-- dotmat1=nn.MM(false,true)
-- dotmat2=nn.MM(false,true)
-- dotrshp=nn.View(-1):setNumInputDims(2) -- reshaping into batch-by-nref matrix for minibatch processing

-- qrshp=rshp:forward(queries);
-- trshp=rshp:forward(targets); --print(trshp)
-- crshp=rshp:forward(confounders); --print(crshp)
-- -- print(qrshp)
-- -- print(trshp)
-- -- print(crshp)

-- tqdot=dotmat1:forward({qrshp,trshp})
-- tqdotr=dotrshp:forward(tqdot)
-- print(tqdot)
-- print(tqdotr)
-- cqdot=dotmat2:forward({qrshp,crshp})
-- cqdotr=dotrshp:forward(cqdot)
-- print(cqdot)
-- print(cqdotr)

-- crit=nn.MarginRankingCriterion(0.1)
