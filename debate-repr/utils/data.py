def ids_to_string(tokenizer, ids):
	return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))

# 給定 post_ids (來自 tokenizer)，以及 adu (text)
# 尋找是否有一條 post_ids 的 slice，可以完全符合 adu_ids
# 有的話回傳 inclusive interval (i, j)
# 否則回傳 None
def get_adu_span_index(tokenizer, post_ids, adu):
	# 尤其使用在如 Roberta 等 sub-word tokenization，
	# substring 的 token list 不一定會在 string 的 token list 中，
	# 因此需要特殊的方式搜尋。
	# 假定 post_ids 轉回文字後，adu 是在 文字內的
	adu_in_post = adu in ids_to_string(tokenizer, post_ids)
	assert adu_in_post
	
	# 搜尋 start & end 區間
	start = 0
	end = len(post_ids)

	for start in range(0, len(post_ids)):
		post_slice = post_ids[start:end]
		adu_in_slice = adu in ids_to_string(tokenizer, post_slice)
		if not adu_in_slice:
			start -= 1
			break

	for end in range(len(post_ids), start, -1):
		post_slice = post_ids[start:end]
		adu_in_slice = adu in ids_to_string(tokenizer, post_slice)
		if not adu_in_slice:
			break
		
	return (start, end)

# lookup is dict[comment_str] -> dict[(int, int)] -> (int, int)
def get_span_from_lookup(tokenizer, comment, adu, lookup):
	comment_text = comment.get_comment()
	adu_text = comment_text[adu[0]:adu[1]]
	if not comment_text in lookup:
		comment_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment_text))
		lookup[comment_text] = { 'ids': comment_ids }
	if not adu in lookup[comment_text]:
		comment_ids = lookup[comment_text]['ids']
		span = get_adu_span_index(tokenizer, comment_ids, adu_text)
		lookup[comment_text][adu] = span
	return lookup[comment_text][adu]
