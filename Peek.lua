--[[
-- useful code by German for debugging purposes

-- example that prints a 3-dimensional tensor (the result of applying a linear operation to 2-dimensional random one):

local i = nn.Linear(2,3)()
local o = nn.Sigmoid()(nn.Peek()(i))

g = nn.gModule({i},{o})

g:forward(torch.rand(2))
--]]

local Peek, parent = torch.class('nn.Peek', 'nn.Module')
function Peek:__init(report_func)
   parent.__init(self)
   self.report_func = report_func
end
 
function Peek:updateOutput(input)
    print('Module input')
    
    self.output=input
    if self.report_func then
      print(self.report_func(input))
    else
      print(input)
    end
      
    return self.output
end
function Peek:updateGradInput(input, gradOutput)
    print('Module grad output')
    print(gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

