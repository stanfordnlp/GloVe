function index = WordLookup_octave(InputString)
global wordMap

if isfield(wordMap, InputString)
  index = wordMap.(InputString);
elseif isfield(wordMap, '<unk>')
  index = wordMap.('<unk>');
else
  index = 0;
end
