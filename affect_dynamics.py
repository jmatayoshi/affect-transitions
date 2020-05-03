import numpy as np


def remove_self_transitions(seq):
    return [st for i, st in enumerate(seq) if i == 0 or st != seq[i - 1]]


def generate_sequence(seq_length, state_dict, base_rates,
                      include_self_transitions=True):
    state_ind = np.arange(len(state_dict))
    if include_self_transitions:
        seq = list(
            np.random.choice(
                list(state_dict.values()),
                size=seq_length,
                p=base_rates
            )
        )
    else:
        base_rates = np.array(base_rates)
        # Construct mapping from each state to its index in state_dict
        inv_state_dict = {}
        for i in range(len(state_dict)):
            inv_state_dict[state_dict[i]] = i
        # Sample first state
        seq = [state_dict[np.random.multinomial(1, base_rates).argmax()]] 
        for i in range(seq_length - 1):
            # Get index of previous state
            prev_ind = inv_state_dict[seq[-1]]
            # Get indices of possible next states
            next_ind = np.setdiff1d(state_ind, [prev_ind])
            # Construct conditional rates of next states
            cond_next_rates = base_rates[next_ind] / (1 - base_rates[prev_ind])
            # Sample next state
            # np.random.multinomial is faster here than np.random.choice
            seq.append(state_dict[next_ind[
                np.random.multinomial(1, cond_next_rates).argmax()]])
    return seq


def get_counts(seq, states):
    next_count = {a: 0 for a in states}
    cond_count = {a: {b: 0 for b in states} for a in states}
    num_tr = len(seq) - 1
    # Compute next and conditional counts
    for i in np.arange(1, len(seq)):
        for a in states:
            if seq[i - 1] == a:
                for b in states:
                    if seq[i] == b:
                        cond_count[a][b] += 1
                        next_count[b] += 1
                        break
                break
    cond_count_list = []
    for a in cond_count:
        cond_count_list.extend(list(cond_count[a].values()))

    return list(next_count.values()), cond_count_list


def get_L_star_vals(a, b, next_counts, cond_counts, use_mean_rates=True):
    num_states = next_counts.shape[1]
    # Column indices where next != a (i.e., transitions in T_{A_complement})
    a_comp_ind = (
        np.array([i for i in range(num_states) if i != a])
    )
    # Count transitions where prev == a and next != a
    a_comp_cond_sum = cond_counts[:, a_comp_ind + a*num_states].sum(axis=1)
    if use_mean_rates:      
        # Compute L_star using base rates averaged over the whole sample
        # of sequences; note that as opposed to the computation of
        # L_star below, we only exclude samples with P(b|a) == nan; that is,
        # we only exclude sequences with no transitions from a to another state
        sample_pos = np.flatnonzero(
            a_comp_cond_sum > 0
        )
        # Compute mean base rate of b restricted to transitions with next != a
        modified_mean_base_rate = np.mean(
            next_counts[sample_pos, b] /
            next_counts[sample_pos, :][:, a_comp_ind].sum(axis=1)
        )
        # Compute conditional rate of b restricted to transitions with next != a
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_comp_cond_sum[sample_pos]
        )
        L_star_vals = (
            (cond_rates - modified_mean_base_rate)
            / (1 - modified_mean_base_rate)
        )
    else:
        # Compute L_star using base rates from each individual sequence

        # Column indices where next != a and next != b
        a_b_comp_ind = (
            np.array([i for i in range(num_states) if i != a and i != b])
        )
        # Count transitions where next != a or next != b
        a_b_comp_sum = next_counts[:, a_b_comp_ind].sum(axis=1)
        # Count transitions where next != a        
        a_comp_sum = next_counts[:, b] + a_b_comp_sum
        # Find samples where:
        #  (a) P(b|a) != nan
        #  (b) P(b) < 1
        sample_pos = np.flatnonzero(
            (a_comp_cond_sum > 0) & (a_b_comp_sum > 0)            
        )        
        # Compute base rates of b restricted to transitions with next != a
        modified_base = (
            next_counts[sample_pos, b] / a_comp_sum[sample_pos]
        )
        # Compute conditional rate of b restricted to transitions with next != a
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_comp_cond_sum[sample_pos]
        )       
        L_star_vals = (
            (cond_rates - modified_base)
            / (1 - modified_base)
        )
    return L_star_vals


def get_L_vals(a, b, next_counts, cond_counts, use_mean_rates=True):
    num_states = next_counts.shape[1]
    # Count transitions where prev == a and next != a
    a_cond_sum = cond_counts[
        :, np.arange(num_states) + a*num_states].sum(axis=1)
    if use_mean_rates:      
        # Compute L using base rates averaged over the whole sample
        # of sequences.  Note that as opposed to the computation of
        # L below, we only exclude samples with P(b|a) == nan; that is,
        # we only exclude sequences with no transitions from a 
        sample_pos = np.flatnonzero(
            a_cond_sum > 0
        )
        # Compute mean base rate of b 
        mean_base_rate = np.mean(
            next_counts[sample_pos, b] /
            next_counts[sample_pos, :].sum(axis=1)
        )
        # Compute conditional rate of b 
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_cond_sum[sample_pos]
        )
        L_vals = (
            (cond_rates - mean_base_rate)
            / (1 - mean_base_rate)
        )
    else:
        # Compute L using base rates from each individual sequence

        # Column indices where next != a and next != b
        b_comp_ind = (
            np.array([i for i in range(num_states) if i != b])
        )
        # Count transitions where next != b
        b_comp_sum = next_counts[:, b_comp_ind].sum(axis=1)
        # Find samples where:
        #  (a) P(b|a) != nan
        #  (b) P(b) < 1
        sample_pos = np.flatnonzero(
            (a_cond_sum > 0) & (b_comp_sum > 0)            
        )        
        # Compute base rates of b
        base_rates = (
            next_counts[sample_pos, b] / next_counts[sample_pos, :].sum(axis=1)
        )
        # Compute conditional rate of b
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_cond_sum[sample_pos]
        )       
        L_vals = (
            (cond_rates - base_rates)
            / (1 - base_rates)
        )
    return L_vals


def compile_sequence_counts(seq_list, states):
    next_counts = []
    cond_counts = []    
    for seq in seq_list:
        count_res = get_counts(seq, states)
        next_counts.append(count_res[0])
        cond_counts.append(count_res[1])
    return np.array(next_counts), np.array(cond_counts)


def print_vals(val_array, state_dict, title):
    print('\n\n' + title + '\n')
    print('Prev\\Next' + '\t' + '\t'.join(map(str, list(state_dict.values()))))
    for i in range(len(state_dict)):
        print(state_dict[i] + '\t\t'
              + '\t'.join(map(str, val_array[i, :].round(4))))
    return


def run_simulations(
        num_trials=50000,
        seq_length=21,
        states=['A', 'B', 'C', 'D'],
        base_rates=np.array([0.45, 0.45, 0.05, 0.05]),
        verbose=True,
        include_self_transitions=True,
        compute_L=True
):
    num_states = len(states)
    state_dict = {}
    for i in range(num_states):
        state_dict[i] = states[i]
    seq_list = []
    reduced_seq_list = []
    for i in range(num_trials):
        seq = generate_sequence(seq_length, state_dict, base_rates,
                    include_self_transitions=include_self_transitions)        
        reduced_seq = remove_self_transitions(seq)

        seq_list.append(seq)
        reduced_seq_list.append(reduced_seq)

    res = []
    # Compute L_star using individual base rates from full sequences with
    # self-transitions included     
    res.append(
        compute_statistic(seq_list, states, L_star=True,
                          use_mean_rates=False)
    )
    # Compute L_star using mean base rates from full sequences with
    # self-transitions included         
    res.append(
        compute_statistic(seq_list, states, L_star=True,
                          use_mean_rates=True)
    )
    if compute_L:
        # Compute L using individual base rates from reduced sequences with
        # self-transitions removed             
        res.append(
            compute_statistic(seq_list, states, L_star=False,
                              use_mean_rates=False)
        )
        # Compute L_star using mean base rates from full sequences with
        # self-transitions included                  
        res.append(
            compute_statistic(seq_list, states, L_star=False,
                              use_mean_rates=True)
        )            

    
    if verbose:
        title_list = [
            'L_star with individual base rates from full sequences',
            'L_star with mean base rates from full sequences',
            'L with individual base rates from reduced sequences',
            'L with mean base rates from reduced sequences',
        ]
        end_ind = 2
        if compute_L:
            end_ind = len(title_list)
        for i in range(end_ind):
            print_vals(res[i], state_dict, title_list[i])
            
    return (
        seq_list,
        reduced_seq_list,
        res
    )


def compute_statistic(seq_list, states, L_star=True,
                      use_mean_rates=True):
    """ General function for computing L_star and L statistics

    Parameters
    ----------
    seq_list : list of lists
        Each entry in the list is a sequence (list) of affective states; note 
        that self-transitions are automatically removed if L_star is false
        Example:
            [
                ['A', 'C', 'C', 'B', 'C'],
                ['B', 'C', 'A', 'C'],
                ['C', 'C', 'C', 'B', 'B', 'A']
            ]
    states : list 
        List containing all possible affective states 
        Example:
            ['A', 'B', 'C']
    L_star : bool, default=True 
        If true compute L_star statistic; otherwise, remove 
        self-transitions and compute L statistic
    use_mean_rates : bool, default=True
        If true compute base rates averaged over all sequences; otherwise,
       compute base rates individually per sequence
    """
    if L_star:
        input_list = seq_list
    else:
        input_list = []
        for i in range(len(seq_list)):
            input_list.append(remove_self_transitions(seq_list[i]))
        
    next_counts, cond_counts = compile_sequence_counts(input_list, states)

    num_states = len(states)
    res = np.full((num_states, num_states), np.nan)
    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                if L_star:
                    res[i, j] = np.mean(
                        get_L_star_vals(i, j, next_counts, cond_counts, 
                                        use_mean_rates=use_mean_rates)
                    )
                else:
                    res[i, j] = np.mean(
                        get_L_vals(i, j, next_counts, cond_counts, 
                                   use_mean_rates=use_mean_rates)
                    )

    return res


def base_rate_analysis(
        states=['A', 'B', 'C', 'D'],
        base_rates=np.ones(4)*0.25,
        num_steps=24,
        rate_step=[0.03, -0.01, -0.01, -0.01],
        num_trials=50000,
        seq_length=21
):
    """ Run numerical experiments 
    
    Experiment 1 parameters: 
        states=['A', 'B', 'C', 'D'], 
        base_rates=np.ones(4)*0.25, 
        num_steps=24, 
        rate_step=[0.03, -0.01, -0.01, -0.01],
        num_trials=50000,
        seq_length=21

    Experiment 2 parameters: 
        states=['A', 'B', 'C', 'D'], 
        base_rates=np.ones(4)*0.25, 
        num_steps=23, 
        rate_step=[0.01, 0.01, -0.01, -0.01],
        num_trials=50000,
        seq_length=21
    
    """
    rate_step = np.array(rate_step)
    indiv_rate_results = []
    mean_rate_results = []
    all_base_rates = []
    for i in range(num_steps):
        sim_res = run_simulations(
            num_trials=num_trials,
            seq_length=seq_length,
            states=states,
            base_rates=base_rates,
            verbose=False,
            include_self_transitions=True,
            compute_L=False
        )
        indiv_rate_results.append(sim_res[2][0])
        mean_rate_results.append(sim_res[2][1])
        all_base_rates.append(list(base_rates))
        if i < num_steps - 1:
            base_rates += rate_step
    return indiv_rate_results, mean_rate_results, all_base_rates
