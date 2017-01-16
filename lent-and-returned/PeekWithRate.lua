local PeekWithRate, parent = torch.class('nn.PeekWithRate', 'nn.Module')

function PeekWithRate:__init(name, print_frequency, print_type)
  self.name = name
  self.print_type = print_type
  self.print_frequency = print_frequency
  self.iteration = -1
end

function PeekWithRate:updateOutput(input)
  self.iteration = self.iteration + 1
  if (self.iteration % self.print_frequency == 0) then
    print(self.name .. " " .. self.iteration)
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

function PeekWithRate:updateGradInput(input, gradOutput)
  return gradOutput
end
