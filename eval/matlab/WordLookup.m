function index = WordLookup(InputString)
global wordMap
if wordMap.isKey(InputString)
    index = wordMap(InputString);
elseif wordMap.isKey('<unk>')
    index = wordMap('<unk>');
else
    index = 0;
end
