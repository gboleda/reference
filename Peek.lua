--[[
-- useful code by German for debugging purposes

-- example that prints a 3-dimensional tensor (the result of applying a linear operation to 2-dimensional random one):

local i = nn.Linear(2,3)()
local o = nn.Sigmoid()(nn.Peek()(i))

g = nn.gModule({i},{o})

g:forward(torch.rand(2))
--]]

local Peek, parent = torch.class('nn.Peek', 'nn.Module')

function Peek:__init(name, print_frequency, print_type)
  self.name = name
  self.print_type = print_type
  self.print_frequency = print_frequency
  self.iteration = -1
end

function Peek:updateOutput(input)
  self.iteration = self.iteration + 1
  if (self.iteration % self.print_frequency == 0) then
    print(self.name)
    local sizeString = input:size(1)
    local num_e = input:size(1)
    if input:dim() > 1 then
      sizeString = sizeString .. "," input:size(2)
      num_e = num_e * input:size(2)
    end
    if input:dim() > 2 then
      sizeString = sizeString .. "," input:size(3)
      num_e = num_e * input:size(3)
    end
    if (self.print_type == 1) then -- print size
      print(sizeString)
    elseif (self.print_type == 2) then--print matrix
      print(input)
    else -- average abs
      print(input:norm() / math.sqrt(num_e))      
    end
  end
  return input
end

function Peek:updateGradInput(input, gradOutput)
    print('Module grad output')
    print(gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end