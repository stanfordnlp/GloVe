addpath('./eval/octave');
if(~exist('vocab_file')) 
    vocab_file = 'vocab.txt';
end
if(~exist('vectors_file')) 
    vectors_file = 'vectors.bin';
end

fid = fopen(vocab_file, 'r');
words = textscan(fid, '%s %f');
fclose(fid);
words = words{1};
vocab_size = length(words);
global wordMap

wordMap = struct();
for i=1:numel(words)
    wordMap.(words{i}) = i;
end

fid = fopen(vectors_file,'r');
fseek(fid,0,'eof');
vector_size = ftell(fid)/16/vocab_size - 1;
frewind(fid);
WW = fread(fid, [vector_size+1 2*vocab_size], 'double')'; 
fclose(fid); 

W1 = WW(1:vocab_size, 1:vector_size); % word vectors
W2 = WW(vocab_size+1:end, 1:vector_size); % context (tilde) word vectors

W = W1 + W2; %Evaluate on sum of word vectors
W = bsxfun(@rdivide,W,sqrt(sum(W.*W,2))); %normalize vectors before evaluation
evaluate_vectors_octave(W);
exit

