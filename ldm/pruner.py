def prune_checkpoint(old_state):
    if int(old_state['global_step']) > 0:
        print(f"Pruning Checkpoint")
        pruned_checkpoint = dict()
        print(f"Checkpoint Keys: {old_state.keys()}")
        for key in old_state.keys():
            if key != "optimizer_states":
                pruned_checkpoint[key] = old_state[key]
        else:
            print("Removing optimizer states from checkpoint")
        if "global_step" in old_state:
            print(f"This is global step {old_state['global_step']}.")
        old_state = pruned_checkpoint['state_dict'].copy()
        new_state = dict()
        for key in old_state:
            new_state[key] = old_state[key].half()
        pruned_checkpoint['state_dict'] = new_state
        return pruned_checkpoint
