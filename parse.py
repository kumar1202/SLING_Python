from __future__ import print_function
import sling
import io



def parse(text,parse_file,model_file):
	
	parser = sling.Parser(model_file)
	doc = parser.parse(text)
	#print (doc.frame.data(pretty=True))

	with io.open(parse_file+".txt","w+") as fp:
		fp.write(str(doc.frame.data(pretty=True)).decode('utf-8'))

	mentions = []
	tokens = []
	for token in doc.tokens:
		tokens.append(token.text)

	for m in doc.mentions:
		mentions.append(doc.phrase(m.begin, m.end))

	return (tokens,mentions)
