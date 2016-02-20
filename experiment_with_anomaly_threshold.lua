cmd = torch.CmdLine()
cmd:option('--step_size',0.001,'determines how granular exploration of threshold will be')
cmd:option('--gold_file','','this file must have an integer in the second (space or tab-delimited) field: if the value is less or equal 0, it is interpreted as an anomalous sample, if it is greater than 0, it is interpreted as a regular one')
cmd:option('--model_guesses_file','','must have a probability in the second (space or tab-delimited) field')
opt = cmd:parse(arg or {})
print(opt)

-- chunks to read files into
BUFSIZE = 2^23 -- 1MB

print('reading the gold file')

local gold_values_table={}

local f = io.input(opt.gold_file)
while true do
   local lines, rest = f:read(BUFSIZE, "*line")
   if not lines then break end
   if rest then lines = lines .. rest .. '\n' end
   -- traversing current chunk line by line
   for current_line in lines:gmatch("[^\n]+") do
      -- the following somewhat cumbersome expression will remove
      -- leading and trailing space, and load all data onto a table
      local current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
      -- we only care about the second field
      table.insert(gold_values_table,current_data[2])
   end
end
f.close()

local gold_values=torch.Tensor(gold_values_table)
gold_values_table=nil

print('reading the model guesses file')

local prob_values_table={}

local f = io.input(opt.model_guesses_file)
while true do
   local lines, rest = f:read(BUFSIZE, "*line")
   if not lines then break end
   if rest then lines = lines .. rest .. '\n' end
   -- traversing current chunk line by line
   for current_line in lines:gmatch("[^\n]+") do
      -- the following somewhat cumbersome expression will remove
      -- leading and trailing space, and load all data onto a table
      local current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
      -- we only care about the second field
      table.insert(prob_values_table,current_data[2])
   end
end
f.close()

local prob_values=torch.Tensor(prob_values_table)
prob_values_table=nil

local ts=torch.range(0,1,opt.step_size)

for i=1,ts:size(1) do 
   -- there should be an easier way to do the following, but I
   -- couldn't find it (and does not behave as expected)
   local tps=torch.eq(torch.add(torch.gt(gold_values,0),torch.ge(prob_values,ts[i])),2)
   local tns=torch.eq(torch.add(torch.le(gold_values,0),torch.lt(prob_values,ts[i])),2)
   print(ts[i]," ",torch.sum(tps+tns)/tps:size(1))
end
