%  /Applications/MATLAB_R2014b.app/bin/matlab -nodesktop -nodisplay -nosplash -nojvm -r output_convnet_guesses

% assumes installation of matconvnet in:
% /Users/marco/Desktop/matconvnet-1.0-beta18/

% also assumes pre-trained model imagenet-vgg-verydeep-19.mat is in
% directory
% UPDATE!!!

% also assumes images are in dir
% UPDATE!!!!

% HARD-CODED file output name:
% convnet_guesses.txt

run /Users/marco/Desktop/matconvnet-1.0-beta18/matlab/vl_setupnn 

% load the pre-trained CNN
net = load('/Users/marco/Desktop/matconvnet-1.0-beta18/imagenet-vgg-verydeep-19.mat') ;

% load images
%imagefiles = dir('/Users/marco/Desktop/matconvnet-1.0-beta18/*.jpg');
imagefiles = dir('/Users/marco/Desktop/trial/*.jpg');

out_file = fopen('convnet_guesses.txt','w');

for im_index=1:length(imagefiles)
%  im = imread(fullfile('/Users/marco/Desktop/matconvnet-1.0-beta18',imagefiles(im_index).name)) ;
  im = imread(fullfile('/Users/marco/Desktop/trial',imagefiles(im_index).name)) ;
  im_ = single(im) ; % note: 0-255 range
  im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
  im_=im_-repmat(net.meta.normalization.averageImage,224);
  res = vl_simplenn(net, im_) ;
  scores = squeeze(gather(res(end).x)) ;
  [bestScore, best] = max(scores) ;
  fprintf(out_file,'%s %s\n',imagefiles(im_index).name,net.meta.classes.description{best});
end

fclose(out_file);

quit
