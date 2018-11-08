from __future__ import print_function
import io
import re

def parse_mentions(filename):
	
	with io.open(filename) as fp:
		
		parse = fp.read()
		parse_tokens = [str(x) for x in parse.split(" ")]
		transition_table = {}
		r = re.compile("^\#([0-9]{1-3}).*")
		all_mentions = False

		while not all_mentions:

			mention_index = int(parse_tokens.index("/s/document/mention:"))
			parse_tokens = parse_tokens[mention_index+1:]
			mention_token_index = int(parse_tokens.index("/s/phrase/begin:"))
			print(mention_index)
			print(parse_tokens)

			try:
				next_mention_index = int(parse_tokens.index("/s/document/mention:"))

			except ValueError:
				all_mentions = True

			finally:
				mention_index_num = parse_tokens[int(mention_token_index)+1][-2:3]
				transition_table[int(mention_index_num[0])] = list(filter(r.match,parse_tokens[mention_index:next_mention_index]))
				#transition_table[int(mention_index_num[0])] = re.findall('(#)([0-9]{1-3})'," ".join(parse_tokens[mention_index:next_mention_index]))

	return transition_table

if __name__ == "__main__":
	print(parse_mentions("as.txt"))