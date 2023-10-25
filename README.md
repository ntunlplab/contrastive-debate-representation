# Contrastive Debate Representation
Contrastively learning participant representations per round in thread-based debates.

# Sequence Tagging & Argument Relation Identification (M704)

This part aims at applying tags on each token of a raw text, effectively identifying argument components as non-overlapping spans of tokens, and predict whether there exists any relations between pairs of such components.

## Sequence Tagging

Refer to the directory `./sequence-tagging/`. The sequence tagging model accepts as inputs in tensors,

1. Properly padded input token ids (`shape=(batch_size, seq_len)`, where `seq_len` does not exceed the max length for the bert model used)
2. Attention masks (for tensor padding, in the same shape as 1.)
3. CRF masks (for tensor padding, and cls/sep tokens, in the same shape as 1.)
4. Ground truth labels in integers (for model training only)

```python
model = Bert_CRF.from_pretrained(
    args.bert_model, 
    bio_num_labels=bio_num_labels, 
    constraints=constraints, 
    include_start_end_transitions=True)

# training
loss, bio_scores, labels_pred = model(
    input_ids=tokens,
    attention_mask=input_masks,
    crf_mask=crf_masks,
    bio_labels=labels,
    check=True
)

# inferencing
labels_pred = model(
    input_ids=input_ids,
    attention_mask=masks,
    crf_mask=crf_masks
)['output']
```

## Argument Relation Identification (ARI)

Refer to the directory `./argument-component-relation/`. The argument component identification model identifies whether two components (token span) in texts are related, and is used under two circumstances, **inner** and **inter**.

For **inner** ARI, the model is given 1 text, and 2 ADUs (components) represented by their indices in the text. The 2 components reside in the same text, so their relation in within the given text (hence **inner**).

For **inter** ARI, the model is instead given 2 texts, and 2 ADUs, residing in the 2 texts respectively. Thus, the relation between the 2 components are across 2 different texts (hence **inter**).

Under both circumstances, the model accepts as inputs in tensors,

1. The text tensors and their masks
    - Generally, there are two such tensors, even for the **inner** case.
    - **inner**: text tensors 1 and 2 are identical.
    - **inter**: text tensors 1 and 2 are distinct.
    - A special case occurs when using Longformer-based models. Such models can usually handle longer data, so we can instead concatenate text tensors 1 and 2, utilizing only one tensor for both texts. (**inner**: `[POST1] text tensor`, **inter**: `[POST1] text tensor 1 [POST2] text tensor 2`)
2. The 2 ADU (component) index tensors (`shape=(batch_size, 2)`) standing for the inclusive starting and ending indices of the argument component represented by its token span.
3. ADU Order (`shape=(batch_size,)`) standing for whether ADU 1 appears before ADU 2, or the other way around.
4. Is Inners (`shape=(batch_size,)`) states whether each data entry is **inner** or **inter**.

```python
model_class = { 
    'bert-base-uncased': BertModel, 
    'Jeevesh8/sMLM-RoBERTa': RobertaModel,
    'Jeevesh8/sMLM-LF': LongformerModel }[args.model_name]

model = Model_Adu_Relation(
    model_class=model_class, 
    model_name=args.model_name, 
    lstm_hidden=args.lstm_hidden_dim, 
    do_inners_embed=args.do_inners_embed, 
    do_adu_order=args.do_adu_order,
    task='link')

# training & inferencing
# feats.shape is (batch_size,), standing for whether there exists a link between the two ADUs
feats = model(post_1_tensors, post_2_tensors, post_1_mask, post_2_mask, adu_1_tensors, adu_2_tensors, adu_orders, is_inners_embed)
```

# Argument Relation Classification (M705)

The argument relation classification (ARC) model works very similarly to the ARI model, including their inputs. Given texts and two ADUs (**inner** or **inter**), the ARC model assumes there exists some relation between them (as predicted by the ARI model), and proceeds to give the *type* of their relation.

Using the models in **M704** and **M705**, the components, relations, and relation types in a given text can be extracted, forming a directed graph of it.

Refer to the directory `./argument-component-relation/`.

```python
model_class = { 
    'bert-base-uncased': BertModel, 
    'Jeevesh8/sMLM-RoBERTa': RobertaModel,
    'Jeevesh8/sMLM-LF': LongformerModel }[args.model_name]

model = Model_Adu_Relation(
    model_class=model_class, 
    model_name=args.model_name, 
    lstm_hidden=args.lstm_hidden_dim, 
    do_inners_embed=args.do_inners_embed, 
    do_adu_order=args.do_adu_order,
    task='link_type')

# training & inferencing
# feats.shape is (batch_size, num_classes), standing for the type of each link
feats = model(post_1_tensors, post_2_tensors, post_1_mask, post_2_mask, adu_1_tensors, adu_2_tensors, adu_orders, is_inners_embed)
```

# Contrastively Learning Text Representations in Threaded Discussion Trees (M706)

Refer to the directory `./debate-repr/`. The contrastive text representation model for threaded debate discussion tree (ContraDebate) takes as inputs a variety of information regarding a **comment** in a discussion tree (presumably in an argumentative setting), and outputs its vector representation using self-supervised learning methods.

Due to the complexities of the data involved, the model deals with one data entry at a time. It taskes as inputs,

1. The tensor of the token ids and masks of the **comment** in interest, and its entire contexts (uproot parents in the tree), forming a list. `shape=(num_comments, seq_len)`.
    - Suppose the root comment **r** is replied to by the comment **c1**, which is in turn replied to by the comment **c2**.
    - Then, the comment ids tensor contains 3 elements (`num_comments=3`)
    - The first looks like `[POST1] token_ids(r)`.
    - The second looks like `[POST1] token_ids(c1) [POST2] token_ids(r)`.
    - The last looks like `[POST1] token_ids(c2) [POST2] token_ids(c1)`.
2. The half-open, token-wise span of the actual comment texts for extracting the LSTM-minus-based embedding of each comment text (usually is `(1, len(comment_ids)-2)` for each comment). `shape=(num_comments, 2)`.
3. The additional *information scores* for each comment. There is no fixed way to construct this feature, but our work used the statistics of the outputs of three pre-trained models tested on the comments ([unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert), [q3fer/distilbert-base-fallacy-classification](https://huggingface.co/q3fer/distilbert-base-fallacy-classification), and [arpanghoshal/EmoRoBERTa](https://huggingface.co/arpanghoshal/EmoRoBERTa)). `shape=(num_comments, custom_info_scores_dim)`.
4. The global attention mask for Longformer-based models, which is only true for `[POST1]` and `[POST2]` tokens.
5. ADU Spans & Masks: The exclusive, token-wise ADU Spans for each comment. Should be properly padded in dimension 1. `shape=(num_comments, max_num_adu_spans, 2)`.
6. Local ADU In Use Masks: The mask for ADU Spans that are (1) local to the comment node and (2) has interaction with other ADUs. `shape=(num_comments, max_num_adu_spans)`
7. Title ADU Span: The exclusive, token-wise span for the main topic ADU in the **root comment** (applicable when dealing with networking forums where all posts have a topic).
8. Inner Pairs: For each comment, there is a list containing tuples of the form `(adu_1_index, adu_2_index, realtion_type)`, where the indices correspond to the elements in ADU Spans. All comments have such a list.
9. Inter Pairs: For each comment, there is a list containing tuples of the form `(adu_1_index, adu_2_index, realtion_type)`, where the indices correspond to the elements in ADU Spans. All comments except the root comment have such a list. `adu_1` is from the comment, and `adu_2` is from the comment being replied to.

The outputs of this model are then used to train against a contrastive learning function.

```python
meta_paths = [
    ['Attack'], ['Support'],
    ['Attack', 'Support'],
    ['Support', 'Attack'],

    ['AttackedBy'], ['SupportedBy'],
    ['AttackedBy', 'SupportedBy'],
    ['SupportedBy', 'AttackedBy'],
]

model = DebateModel(meta_paths=meta_paths, info_scores_dim=len(utils.info_all_keys))

# training: only returns the representation of the last comment given (comment_ids[-1])
comment_repr = model(
    None,
    comment_ids, comment_mask, comment_spans, info_scores,
    global_attention_mask,
    adu_spans, adu_masks, local_adu_in_use_masks, title_adu_span_index,
    None, inner_pairs, inter_pairs, device, return_all=False)

# inferencing: returns the representations of all comments, including the one in interest, and all its contexts
comment_repr = model(
    None,
    comment_ids, comment_mask, comment_spans, info_scores,
    global_attention_mask,
    adu_spans, adu_masks, local_adu_in_use_masks, title_adu_span_index,
    None, inner_pairs, inter_pairs, device, return_all=True)
```
