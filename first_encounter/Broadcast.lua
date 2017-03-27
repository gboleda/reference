local Broadcast, parent = torch.class('nn.Broadcast', 'nn.Module')

function Broadcast:__init(outputSize)
   parent.__init(self)
   self.outputSize = outputSize
   self.tmp = torch.Tensor()
end

function Broadcast:updateOutput(input)
   if input:dim() == 2 then
       self.tmp:resize(input:size(1), 1, input:size(2))
       self.tmp:copy(input)
       self.output:resize(input:size(1), self.outputSize, input:size(2))
       self.output:copy(self.tmp:expandAs(self.output))
   else
      error('input must be matrix')
   end
   return self.output
end

function Broadcast:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(input)
      self.gradInput:copy(gradOutput:sum(2))
      return self.gradInput
   end
end