-- preambles

cmd = torch.CmdLine()
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)')
cmd:option('--word_embedding_size',400,'dimensionality of embeddings')
cmd:option('--word_count',2000,'number of words in embeddings file')
cmd:option('--output_word_dot_products_file','','to print word embedding dot product matrix')
opt = cmd:parse(arg or {})
print(opt)

-- other general parameters
-- chunks to read files into
BUFSIZE = 2^23 -- 1MB

print('reading the data')

local embeddings=torch.Tensor(opt.word_count,opt.word_embedding_size)
local current_data={}
local f = io.input(opt.word_embedding_file)
local line_counter=0
while true do
   local lines, rest = f:read(BUFSIZE, "*line")
   if not lines then break end
   if rest then lines = lines .. rest .. '\n' end
   -- traversing current chunk line by line
   for current_line in lines:gmatch("[^\n]+") do
      line_counter=line_counter+1
      -- the following somewhat cumbersome expression will remove
      -- leading and trailing space, and load all data onto a table
      current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
      -- first field is id, other fields are embedding vector
      raw_embeddings=
	 torch.Tensor({unpack(current_data,2,#current_data)})
      -- normalize
      embeddings[line_counter]= raw_embeddings/torch.norm(raw_embeddings)
   end
end
f.close()



local similarity_matrix = embeddings*embeddings:t()  

embeddings=nil

local f = io.open(opt.output_word_dot_products_file,"w")

for i=1,similarity_matrix:size(1) do
   for j=1,similarity_matrix:size(2) do
      if j==1 then
	 f:write(similarity_matrix[i][j])
      else
	 f:write(" ",similarity_matrix[i][j])
      end
   end
   f:write("\n")
end

f:flush()
f:close()

similarity_matrix=nil

print('all done')

