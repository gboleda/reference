local Peek, parent = torch.class('nn.Peek', 'nn.Module')

function Peek:__init(name, print_matrix)
  self.name = name
  self.print_matrix = print_matrix
end

function Peek:updateOutput(input)
  print(self.name)
  local sizeString = input:size(1)
  if input:dim() > 1 then
    sizeString = sizeString .. "," input:size(2)
  end
  if input:dim() > 2 then
    sizeString = sizeString .. "," input:size(3)
  end
  
  if (self.print_matrix) then
    print(input)
  else
    print(sizeString)
  end
  return input
end

function Peek:updateGradInput(input, gradOutput)
	return gradOutput
end