local BroadCast, parent = torch.class('nn.BroadCast', 'nn.Module')

function BroadCast:__init()
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function BroadCast:updateOutput(input)
  assert(#input == 2, 'input must be a pair of minibatch matrices')
  local a, b = table.unpack(input)
  assert(a:nDimension() == 2 or a:nDimension() == 3, 'first input tensor must be 2D or 3D')
  assert(b:nDimension() == 3, 'second input tensor must be 3D')
  if (a:nDimension() == 2) then
    a = a:view(b:size(1),1,b:size(3))
  end
  local broadcast_a = a:expandAs(b)
  self._output = self._output or broadcast_a.new()
  self._output:resizeAs(broadcast_a):copy(broadcast_a)
  return self._output
end

function BroadCast:updateGradInput(input, gradOutput)
  local v1  = input[1]
  local v2  = input[2]
 
  if #self.gradInput ~= 2 then
     self.gradInput[1] = self.gradInput[1] or v1.new()
     self.gradInput[2] = self.gradInput[2] or v1.new()
  end
  self.gradInput[2]:resizeAs(v2):fill(0)
  self.gradInput[1] = gradOutput:sum(2) 
  if v1:dim() == 2 then
     self.gradInput[1] = self.gradInput[1]:view(v1:size(1),v1:size(2))
  end
  return self.gradInput
end