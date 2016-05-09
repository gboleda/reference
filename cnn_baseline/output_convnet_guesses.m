%  /mnt/8tera/apps/matlabR2015b/bin/matlab -nodesktop -nodisplay -nosplash -nojvm -r output_convnet_guesses

% assumes we can rely on Ravi\'s installation of matconvnet in:
% 

% also assumes pre-trained model imagenet-vgg-verydeep-19.mat is in
% current directory

% also assumes images are in dir
% images

% HARD-CODED file output name:
% newest_convnet_guesses.txt

run  /home/ravi.shekhar/matconvnet-1.0-beta17/matlab/vl_setupnn

% load the pre-trained CNN
net = load('imagenet-vgg-verydeep-19.mat') ;

% load images
imagefiles = dir('images/*.jpg');

out_file = fopen('newest_convnet_guesses.txt','w');

for im_index=1:length(imagefiles)
  im = imread(fullfile('images',imagefiles(im_index).name));

  im_ = single(im) ; % note: 0-255 range
  im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
  im_=im_-repmat(net.meta.normalization.averageImage,224);
  res = vl_simplenn(net, im_) ;
  scores = squeeze(gather(res(end).x)) ;
  [bestScore, best] = max(scores) ;
  fprintf(out_file,'%s\t%s\n',imagefiles(im_index).name,net.meta.classes.description{best});
end

fclose(out_file);

quit
