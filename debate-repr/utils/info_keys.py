info_emotion_keys = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
info_toxicity_keys = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
info_logic_keys = ["ad hominem", "ad populum", "appeal to emotion", "circular reasoning", "equivocation", "fallacy of credibility", "fallacy of extension", "fallacy of logic", "fallacy of relevance", "false causality", "false dilemma", "faulty generalization", "intentional", "miscellaneous"]

info_all_keys = \
    ['e_c_' + key for key in info_emotion_keys] + \
    ['e_m_' + key for key in info_emotion_keys] + \
    ['e_a_' + key for key in info_emotion_keys] + \
    ['e_v_' + key for key in info_emotion_keys] + \
    ['t_m_' + key for key in info_toxicity_keys] + \
    ['t_a_' + key for key in info_toxicity_keys] + \
    ['t_v_' + key for key in info_toxicity_keys] + \
    ['l_m_' + key for key in info_logic_keys] + \
    ['l_a_' + key for key in info_logic_keys] + \
    ['l_v_' + key for key in info_logic_keys]

    # ['l_c_' + key for key in info_logic_keys] + \
