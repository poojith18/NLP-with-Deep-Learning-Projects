from state import Token, ParseState, shift, left_arc, right_arc, is_final_state


def get_deps(words_lists, actions, cwindow):
    """ Computes all the dependencies set for all the sentences according to 
    actions provided
    Inputs
    -----------
    words_lists: List[List[str]].  This is a list of lists. Each inner list is a list of words in a sentence,
    actions: List[List[str]]. This is a list of lists where each inner list is the sequence of actions
                Note that the elements should be valid actions as in `tagset.txt`
    cwindow: int. Context window. Default=2
    """
    all_deps = []   # List of List of dependencies
    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Intialize stack and buffer appropriately
        stack = [Token(idx=-i-1, word="[NULL]", pos="NULL") for i in range(cwindow)]
        parser_buff = []
        for ix in range(len(words_list)):
            parser_buff.append(Token(idx=ix, word=words_list[ix], pos="NULL"))
        parser_buff.extend([Token(idx=ix+i+1, word="[NULL]",pos="NULL") for i in range(cwindow)])
        # Initilaze the parse state
        state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])

        # Iterate over the actions and do the necessary state changes
        for action in actions[w_ix]:
            if action == "SHIFT":
                shift(state)
            elif action[:8] == "REDUCE_L":
                left_arc(state, action[9:])
            else:
                right_arc(state, action[9:])
        assert is_final_state(state,cwindow)    # Check to see that the parse is complete
        right_arc(state, "root")    # Add te root dependency for the remaining element on stack
        all_deps.append(state.dependencies.copy())  # Copy over the dependenices found
    return all_deps
        



def compute_metrics(words_lists, gold_actions, pred_actions, cwindow=2):
    """ Computes the UAS and LAS metrics given list of words, gold and predicted actions.
    Inputs
    -------
    word_lists: List[List[str]]. This is a list of lists. Each inner list is a list of words in a sentence,
    gold_action: List[List[str]]. This is a list of lists where each inner list is the sequence of gold actions
                Note that the elements should be valid actions as in `tagset.txt`
    pred_action: List[List[str]]. This is a list of lists where each inner list is the sequence of predicted actions
                Note that the elements should be valid actions as in `tagset.txt`
 
    Outputs
    -------
    uas: int. The Unlabeled Attachment Score
    las: int. The Lableled Attachment Score
    """
    lab_match = 0  # Counter for computing correct head assignment and dep label
    unlab_match = 0 # Counter for computing correct head assignments
    total = 0       # Total tokens

    # Get all the dependencies for all the sentences
    gold_deps = get_deps(words_lists, gold_actions, cwindow)    # Dep according to gold actions
    pred_deps = get_deps(words_lists, pred_actions, cwindow)    # Dep according to predicted actions

    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Iterate over words in a sentence
        for ix, word in enumerate(words_list):
            # Check what is the head of the word in the gold dependencies and its label
            for dep in gold_deps[w_ix]:
                if dep.target.idx == ix:
                    gold_head_ix = dep.source.idx
                    gold_label = dep.label
                    break
            # Check what is the head of the word in the predicted dependencies and its label
            for dep in pred_deps[w_ix]:
                if dep.target.idx == ix:
                    # Do the gold and predicted head match?
                    if dep.source.idx == gold_head_ix:
                        unlab_match += 1
                        # Does the label match? 
                        if dep.label == gold_label:
                            lab_match += 1
                    break
            total += 1

    return unlab_match/total, lab_match/total

